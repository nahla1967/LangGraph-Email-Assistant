from typing import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
model = ChatOllama(
    model="qwen2.5:7b",
    temperature=0
)
class EmailState(TypedDict , total=False):
    email: str
    email_category: str
    spam_reason: str
    is_spam: bool
    email_draft: str
    messages: list

def read_email(state: EmailState) -> dict:
    email = state["email"]

    print("Reading the email...")
    print(f"Sender: {email['sender']}")
    print(f"Subject: {email['subject']}")

    return {}
def classify_email(state: EmailState) -> dict:
    email = state["email"]

    prompt = f"""
    Analyze this email.

    Sender: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}

    Determine whether it is spam or legitimate.

    If it is spam, explain the reason.
    If it is legitimate, choose a category:
    inquiry, complaint, thank you, request, information.

    Return your answer in this format:

    SPAM: yes or no
    REASON: explanation or none
    CATEGORY: category or none
    """

    response = model.invoke(prompt)
    response_text = response.content.lower()

    is_spam = "spam: yes" in response_text

    spam_reason = None
    if is_spam and "reason:" in response_text:
        spam_reason = response_text.split("reason:")[1].split("category:")[0].strip()

    email_category = None
    if not is_spam and "category:" in response_text:
        email_category = response_text.split("category:")[1].strip()

    print("\nClassification result:")
    print(response.content)

    return {
    "is_spam": is_spam,
    "spam_reason": spam_reason,
    "email_category": email_category,
    "messages": state.get("messages", []) + [
        {"role": "assistant", "content": response.content}
    ]
}
def route_email(state: EmailState) -> str:

    if state["is_spam"]:
        return "spam"

    else:
        return "legitimate"
    
def handle_spam(state: EmailState) -> dict:
    print("This email is spam.")
    print("Reason:", state["spam_reason"])

    return {}

def draft_response(state: EmailState) -> dict:
    email = state["email"]
    category = state["email_category"] or "general"

    prompt = f"""
    Draft a polite and professional reply to this email.

    Sender: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}

    Category: {category}

    Keep the response brief and professional.
    """

    response = model.invoke(prompt)

    return {
        "email_draft": response.content
    }
def notify_user(state: EmailState) -> dict:
    print("\nEmail notification")
    print("Category:", state["email_category"])
    print("\nDraft response:")
    print(state["email_draft"])

    return {}

email_graph = StateGraph(EmailState)

email_graph.add_node("read_email", read_email)
email_graph.add_node("classify_email", classify_email)
email_graph.add_node("handle_spam", handle_spam)
email_graph.add_node("draft_response", draft_response)
email_graph.add_node("notify_user", notify_user)

email_graph.add_edge(START, "read_email")

email_graph.add_edge("read_email", "classify_email")

email_graph.add_conditional_edges(
    "classify_email",
    route_email,
    {
        "spam": "handle_spam",
        "legitimate": "draft_response"
    }
)

email_graph.add_edge("handle_spam", END)

email_graph.add_edge("draft_response", "notify_user")

email_graph.add_edge("notify_user", END)

compiled_graph = email_graph.compile()
print(compiled_graph.get_graph().draw_mermaid())

legitimate_email = {
    "sender": "john.smith@example.com",
    "subject": "Meeting request",
    "body": "Hello, I would like to schedule a meeting tomorrow."
}

print("\nStarting LangGraph workflow...")

result = compiled_graph.invoke({
    "email": legitimate_email,
    "messages": []
})

print("\nFinal State:")
print(result)

spam_email = {
    "sender": "winner@lottery-intl.com",
    "subject": "YOU WON $5,000,000!",
    "body": "You won a lottery prize! Send your bank details and pay $100 to claim it."
}

print("\nProcessing spam email...")

spam_result = compiled_graph.invoke({
    "email": spam_email,
    "messages": []
})

print("\nSpam classification:", spam_result["is_spam"])


