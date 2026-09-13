"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import sys
import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## 1. PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm, dịch vụ (VinFast, Vinpearl) và hỗ trợ khách hàng Vingroup.
- Giọng điệu: Chuyên nghiệp, lịch sự, thân thiện, đồng cảm và chính xác.

## 2. AVAILABLE TOOLS
- `search_product_catalog(category, max_price)`: Tra cứu danh mục sản phẩm/dịch vụ xe điện (xe_dien) hoặc du lịch nghỉ dưỡng (du_lich) với mức giá tối đa tính bằng VNĐ.
- `submit_support_ticket(customer_name, issue_description, priority)`: Ghi nhận yêu cầu hỗ trợ, phản ánh sự cố hoặc khiếu nại của khách hàng vào hệ thống ticket.

## 3. CORE RULES
- KHÔNG BAO GIỜ bịa đặt hoặc đoán thông tin về giá bán, thông số kỹ thuật sản phẩm, dịch vụ.
- BẮT BUỘC gọi tool `search_product_catalog` khi người dùng hỏi về danh mục, giá cả hoặc tìm kiếm sản phẩm.
- BẮT BUỘC gọi tool `submit_support_ticket` khi người dùng báo lỗi, sự cố hoặc yêu cầu tạo ticket hỗ trợ.
- Đối với câu hỏi chính sách chung (FAQ) đã biết rõ như chính sách bảo hành, có thể trả lời trực tiếp mà không cần gọi tool.

## 4. OPERATIONAL BOUNDARIES
- Chỉ hỗ trợ thông tin và dịch vụ liên quan đến hệ sinh thái Vingroup (VinFast, Vinpearl, Vinhomes,...).
- Từ chối lịch sự nếu khách hàng yêu cầu các nội dung ngoài phạm vi dịch vụ của Vingroup.

## 5. OUTPUT CONTRACT
- Tuân thủ cấu trúc ReAct:
  - Thought: Phân tích yêu cầu của người dùng và quyết định hành động tiếp theo.
  - Action: Tên công cụ và tham số cần thực thi (nếu cần).
  - Observation: Kết quả nhận được từ công cụ.
  - Final Answer: Câu trả lời hoàn chỉnh bằng Tiếng Việt, chính xác và đầy đủ gửi tới người dùng.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        # Trả về câu trả lời tĩnh (mock) hoặc gọi Gemini API 1 lượt (không dùng tool)
        # Mục tiêu: Quan sát hiện tượng bịa thông tin (hallucination)
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def _detect_intents(self, user_input: str) -> Dict[str, Any]:
        """Phân tích intent độc lập từ user_input."""
        lower_input = user_input.lower()

        # Kiểm tra intent ticket độc lập
        ticket_keywords = [
            "bị lỗi", "lỗi", "sự cố", "hỏng", "ẩm mốc",
            "khiếu nại", "phản hồi", "ticket", "vấn đề nghiêm trọng", "cần xử lý gấp"
        ]
        needs_ticket = any(k in lower_input for k in ticket_keywords)

        # Kiểm tra intent FAQ độc lập
        faq_keywords = ["chính sách", "bao lâu",
                        "quy định", "thế nào", "là gì"]
        is_faq = any(
            k in lower_input for k in faq_keywords) and not needs_ticket

        # Kiểm tra intent catalog độc lập
        catalog_keywords = [
            "xem", "mua", "giá", "dưới", "triệu", "tỷ",
            "resort", "vinpearl", "khách sạn", "du lịch", "có xe", "sản phẩm", "catalog"
        ]
        needs_catalog = any(
            k in lower_input for k in catalog_keywords) and not is_faq

        return {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "is_faq": is_faq
        }

    def _extract_catalog_args(self, user_input: str) -> Dict[str, Any]:
        """Trích xuất tham số cho search_product_catalog."""
        lower_input = user_input.lower()
        if any(w in lower_input for w in ["resort", "du lịch", "du_lich", "khách sạn", "phòng"]):
            category = "du_lich"
        else:
            category = "xe_dien"

        price_match = re.search(
            r'(?:dưới|tối đa|khoảng|<|<=)?\s*(\d+(?:[\.,]\d+)?)\s*(triệu|tỷ|tr|ty|vnd|vnđ|đ)', user_input, re.IGNORECASE)
        if price_match:
            val = float(price_match.group(1).replace(",", "."))
            unit = (price_match.group(2) or "").lower()
            if "tỷ" in unit or "ty" in unit:
                max_price = int(val * 1_000_000_000)
            elif "triệu" in unit or "tr" in unit:
                max_price = int(val * 1_000_000)
            else:
                max_price = int(val)
        else:
            max_price = 999999999999

        return {"category": category, "max_price": max_price}

    def _extract_ticket_args(self, user_input: str) -> Dict[str, Any]:
        """Trích xuất tham số cho submit_support_ticket."""
        # 1. Customer Name
        name_match = re.search(
            r'(?:tôi tên là|tôi tên|tên tôi là|tên là)\s*[:,\s]*([A-ZÀ-Ỵ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỵ][a-zà-ỹ]+)+)', user_input)
        if not name_match:
            name_match = re.search(
                r'(?:tôi tên là|tôi tên|tên tôi là|tên là)\s*[:,\s]*([^,\.\;]+)', user_input, re.IGNORECASE)
        customer_name = name_match.group(
            1).strip() if name_match else "Khách hàng"

        # 2. Issue Description
        issue_match = re.search(
            r'((?:xe|phòng|sản phẩm|dịch vụ)[^,\.]*(?:bị lỗi|lỗi|ẩm mốc|hỏng|sự cố)[^,\.]*)', user_input, re.IGNORECASE)
        if issue_match:
            issue_description = issue_match.group(1).strip()
        else:
            parts = [p.strip() for p in re.split(r'[,;\.]', user_input) if any(
                k in p.lower() for k in ["lỗi", "hỏng", "ẩm mốc", "sự cố"])]
            issue_description = parts[0] if parts else user_input

        # 3. Priority
        if re.search(r'(gấp|nghiêm trọng|khẩn cấp|high)', user_input, re.IGNORECASE):
            priority = "high"
        elif re.search(r'(thấp|nhẹ|low|không vội)', user_input, re.IGNORECASE):
            priority = "low"
        else:
            priority = "medium"

        return {
            "customer_name": customer_name,
            "issue_description": issue_description,
            "priority": priority
        }

    def _build_plan(self, intents: Dict[str, Any], user_input: str) -> List[Dict[str, Any]]:
        """Xây dựng danh sách tool cần thực thi dựa trên intents."""
        plan = []
        if intents.get("needs_catalog"):
            plan.append({
                "tool": "search_product_catalog",
                "args": self._extract_catalog_args(user_input)
            })
        if intents.get("needs_ticket"):
            plan.append({
                "tool": "submit_support_ticket",
                "args": self._extract_ticket_args(user_input)
            })
        return plan

    def _generate_faq_answer(self, user_input: str) -> str:
        """Sinh câu trả lời cho các câu hỏi FAQ."""
        if re.search(r'(bảo hành|pin)', user_input, re.IGNORECASE):
            return (
                "Chính sách bảo hành pin xe điện VinFast kéo dài lên đến 10 năm "
                "(hoặc không giới hạn số km tùy theo dòng xe). VinFast cam kết bảo hành "
                "đổi mới hoặc sửa chữa miễn phí nếu dung lượng pin giảm dưới 70% trong thời hạn bảo hành."
            )
        return (
            "Xin chào quý khách! Tôi là VinAssistant, trợ lý AI chính thức của hệ sinh thái Vingroup. "
            "Tôi có thể hỗ trợ quý khách tra cứu sản phẩm xe điện VinFast, kỳ nghỉ Vinpearl "
            "hoặc tiếp nhận các yêu cầu hỗ trợ kỹ thuật và khiếu nại."
        )

    def _synthesize_answer(self, observations: List[Dict[str, Any]], user_input: str) -> str:
        """Tổng hợp câu trả lời từ các kết quả tool."""
        answers = []
        for obs_item in observations:
            tool_name = obs_item["tool"]
            result = obs_item["result"]

            if tool_name == "search_product_catalog":
                if not result or len(result) == 0:
                    answers.append(
                        "Rất tiếc, không tìm thấy sản phẩm phù hợp với yêu cầu của bạn.")
                else:
                    lines = [
                        "Dưới đây là các sản phẩm phù hợp trong hệ sinh thái Vingroup:"]
                    for p in result:
                        lines.append(
                            f"- {p['name']}: Giá {p['price_vnd']:,} VNĐ. {p.get('description', '')}")
                    answers.append("\n".join(lines))

            elif tool_name == "submit_support_ticket":
                ticket_id = result.get("ticket_id", "")
                customer_name = result.get("customer_name", "quý khách")
                priority = result.get("priority", "medium")
                answers.append(
                    f"Yêu cầu hỗ trợ của quý khách {customer_name} đã được ghi nhận thành công "
                    f"với mã ticket {ticket_id} (Mức độ ưu tiên: {priority}). "
                    f"Đội ngũ kỹ thuật sẽ liên hệ và hỗ trợ trong thời gian sớm nhất."
                )

        return "\n\n".join(answers)

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        self.trace.append({"step": "init", "user_input": user_input})

        intents = self._detect_intents(user_input)
        plan = self._build_plan(intents, user_input)
        observations: List[Dict[str, Any]] = []

        iteration = 1
        while iteration <= self.max_iterations:
            # Case 1: FAQ / Trả lời trực tiếp khi không có tool nào
            if intents.get("is_faq") or (len(plan) == 0 and len(observations) == 0):
                answer = self._generate_faq_answer(user_input)
                self.trace.append({
                    "iteration": iteration,
                    "thought": "Câu hỏi là thông tin chung/FAQ, trả lời trực tiếp mà không cần gọi tool.",
                    "action": "none",
                    "observation": "Direct response generated.",
                    "final_answer": answer
                })
                return {
                    "answer": answer,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed"
                }

            # Case 2: Chỉ cần 1 tool -> thực thi và trả về kết quả trong iteration 1
            if len(plan) == 1 and iteration == 1:
                action = plan.pop(0)
                tool_name = action["tool"]
                tool_args = action["args"]
                tool_func = TOOL_MAP[tool_name]
                obs = tool_func(**tool_args)
                observations.append(
                    {"tool": tool_name, "args": tool_args, "result": obs})
                self.trace.append({
                    "iteration": iteration,
                    "thought": f"Cần gọi tool {tool_name} với tham số: {tool_args}",
                    "action": {"tool": tool_name, "args": tool_args},
                    "observation": obs
                })
                answer = self._synthesize_answer(observations, user_input)
                self.trace.append({"step": "final_answer", "answer": answer})
                return {
                    "answer": answer,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed"
                }

            # Case 3: Nhiều tools -> thực thi từng tool theo từng iteration
            if len(plan) > 0:
                action = plan.pop(0)
                tool_name = action["tool"]
                tool_args = action["args"]
                tool_func = TOOL_MAP[tool_name]
                obs = tool_func(**tool_args)
                observations.append(
                    {"tool": tool_name, "args": tool_args, "result": obs})
                self.trace.append({
                    "iteration": iteration,
                    "thought": f"Thực hiện bước {iteration}: gọi tool {tool_name} với tham số {tool_args}",
                    "action": {"tool": tool_name, "args": tool_args},
                    "observation": obs
                })
                iteration += 1
                continue

            # Case 4: Đã thực thi xong toàn bộ tools, tổng hợp Final Answer
            answer = self._synthesize_answer(observations, user_input)
            self.trace.append({
                "iteration": iteration,
                "thought": "Đã hoàn thành các bước gọi tool. Tiến hành tổng hợp câu trả lời cuối cùng.",
                "step": "final_answer",
                "answer": answer
            })
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": iteration,
                "status": "completed"
            }

        return {
            "answer": "Lỗi: Vượt quá số bước tối đa.",
            "trace": self.trace,
            "iterations": iteration,
            "status": "max_iterations_reached"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import os
    agent = ToolCallingAgent(max_iterations=5)

    # 1. Chế độ Chat tương tác: python template.py --chat
    if len(sys.argv) > 1 and sys.argv[1] == "--chat":
        print("=== CHẾ ĐỘ CHAT TƯƠNG TÁC VỚI VINASSISTANT (gõ 'exit' để thoát) ===")
        while True:
            try:
                user_query = input("\n[Bạn]: ").strip()
                if not user_query:
                    continue
                if user_query.lower() in ["exit", "quit", "q"]:
                    print("Tạm biệt!")
                    break
                res = agent.run(user_query)
                print(f"[VinAssistant]:\n{res['answer']}")
                print(
                    f"-> Số bước (iterations): {res['iterations']} | Trạng thái: {res['status']}")
            except (KeyboardInterrupt, EOFError):
                print("\nĐã thoát.")
                break
        return

    # 2. Chế độ chạy toàn bộ test cases: python template.py --all
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        queries_file = os.path.join(os.path.dirname(
            __file__), "..", "raw-data", "customer_queries.json")
        if os.path.exists(queries_file):
            with open(queries_file, "r", encoding="utf-8") as f:
                queries = json.load(f)
            for idx, q in enumerate(queries, 1):
                print(
                    f"\n{'='*15} Test Case {idx}: {q['id']} ({q['category']}) {'='*15}")
                print(f"❓ Câu hỏi: {q['query']}")
                res = agent.run(q['query'])
                print(f"🤖 Trả lời:\n{res['answer']}")
                print(
                    f"⚙️ Số bước: {res['iterations']} | Trạng thái: {res['status']}")
        return

    # 3. Chế độ mặc định hoặc truyền câu hỏi trực tiếp: python template.py "câu hỏi của bạn"
    user_query = sys.argv[1] if len(
        sys.argv) > 1 else "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
