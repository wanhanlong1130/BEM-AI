import asyncio
import streamlit as st
from automa_ai.client.simple_client import (
    SimpleClient,
)  # assuming your file is named simple_client.py

A2A_SERVER_URL = "http://localhost:10001"


# Cache the client instance
@st.cache_resource
def get_client():
    return SimpleClient(agent_url=A2A_SERVER_URL)


async def send_message_async(user_message: str, context_id: str | None = None):
    client = get_client()
    response_chunks = []
    async for chunk in client.send_streaming_message(user_message, context_id):
        response_chunks.append(chunk)
        yield chunk


def main():
    st.set_page_config(page_title="Automa AI Chat", page_icon="*", layout="centered")
    st.title("Automa AI Chat Interface")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Type your message..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Streaming assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            async def process_stream():
                nonlocal full_response
                with st.spinner("Thinking..."):
                    async for chunk in send_message_async(prompt, st.session_state.get("context_id")):
                        print(chunk)
                        text_part = None

                        ## Case 1: A2A JSON-RPC result object
                        if isinstance(chunk, dict) and "result" in chunk:
                            result = chunk.get("result", {})
                            kind = result.get("kind")
                            context_id = result.get("contextId")
                            if context_id:
                                st.session_state["context_id"] = context_id

                            # === Handle artifact-update ===
                            if kind == "artifact-update":
                                artifact = result.get("artifact", {})
                                parts = artifact.get("parts", [])
                                text_fragments = [
                                    p.get("text") for p in parts if p.get("kind") == "text" and p.get("text")
                                ]
                                if text_fragments:
                                    text_part = "\n".join(text_fragments)
                                    full_response += f"\n\n **Artifact Update**\n{text_part}"
                                    message_placeholder.markdown(full_response + "▌")
                            # === Handle status-update ===
                            if kind == "status-update":
                                status = result.get("status", {})
                                state = status.get("state")
                                message = status.get("message", {})
                                parts = message.get("parts", [])

                                # Extract text fragments (agent response or question)
                                text_fragments = [
                                    p.get("text")
                                    for p in parts
                                    if p.get("kind") == "text" and p.get("text")
                                ]
                                if text_fragments:
                                    text_part = "\n".join(text_fragments)

                                # Handle 'input-required' state
                                if state == "input-required":
                                    st.session_state["awaiting_input"] = True
                                    full_response += (
                                        f"\n\n *Agent is waiting for your response...*\n\n"
                                        f"**Question:** {text_part}"
                                    )
                                    message_placeholder.markdown(full_response)
                                    break  # Stop streaming to wait for user input
                                else:
                                    st.session_state["awaiting_input"] = False
                        ## Case 2: fallback streaming types
                        elif "delta" in chunk and "text" in chunk["delta"]:
                            text_part = chunk["delta"]["text"]
                        elif "message" in chunk and "text" in chunk["message"]:
                            text_part = chunk["message"]["text"]
                        elif "content" in chunk:
                            text_part = chunk["content"]
                        elif "data" in chunk:
                            text_part = chunk["data"]

                        # --- Render text incrementally ---
                        if text_part and state != "input-required":
                            full_response += text_part
                            message_placeholder.markdown(full_response + "▌")

            asyncio.run(process_stream())

        st.session_state["messages"].append(
            {"role": "assistant", "content": full_response}
        )


if __name__ == "__main__":
    main()
#streamlit run examples/code_gbxml/streamlit_ui.py
#what is the required lighting wattage of my gbxml file at .\examples\code_gbxml\gbxml_file.xml living_unit1_Space for CEZ_IECC2021
#what is the area of my gbxml file at .\examples\code_gbxml\SF_model_Output_OS.xml "living_unit1_Space" "ceiling_unit1_Reversed"