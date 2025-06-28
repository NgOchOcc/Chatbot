import streamlit as st
import requests
import json
from typing import List, Dict
import time
import tiktoken

# Cấu hình trang
st.set_page_config(
    page_title="vLLM Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
    .sidebar .sidebar-content {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)

class vLLMClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        # Khởi tạo tokenizer để đếm token
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Đếm số token trong text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: ước tính 1 token ≈ 4 ký tự
            return len(text) // 4
    
    def get_models(self) -> List[str]:
        """Lấy danh sách models có sẵn"""
        try:
            response = requests.get(f"{self.base_url}/v1/models", headers=self.headers)
            if response.status_code == 200:
                models = response.json()
                return [model["id"] for model in models.get("data", [])]
            return []
        except Exception as e:
            st.error(f"Không thể kết nối đến vLLM server: {e}")
            return []
    
    def chat_completion(self, messages: List[Dict], model: str, **kwargs) -> tuple:
        """Gửi yêu cầu chat completion và trả về (response, token_count)"""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000),
            "top_p": kwargs.get("top_p", 0.9),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
            "presence_penalty": kwargs.get("presence_penalty", 0.0),
            "stream": kwargs.get("stream", False)
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                token_count = self.count_tokens(content)
                return content, token_count
            else:
                error_msg = f"Lỗi API: {response.status_code} - {response.text}"
                return error_msg, 0
                
        except Exception as e:
            error_msg = f"Lỗi kết nối: {str(e)}"
            return error_msg, 0
    
    def stream_chat_completion(self, messages: List[Dict], model: str, **kwargs):
        """Streaming chat completion"""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000),
            "top_p": kwargs.get("top_p", 0.9),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
            "presence_penalty": kwargs.get("presence_penalty", 0.0),
            "stream": True
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=60
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]  # Bỏ "data: "
                            if data.strip() == '[DONE]':
                                break
                            try:
                                json_data = json.loads(data)
                                delta = json_data["choices"][0]["delta"]
                                if "content" in delta:
                                    yield delta["content"]
                            except json.JSONDecodeError:
                                continue
            else:
                yield f"Lỗi API: {response.status_code}"
                
        except Exception as e:
            yield f"Lỗi kết nối: {str(e)}"

def init_session_state():
    """Khởi tạo session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vllm_client" not in st.session_state:
        st.session_state.vllm_client = None
    if "models" not in st.session_state:
        st.session_state.models = []
    if "current_model" not in st.session_state:
        st.session_state.current_model = None
    if "total_tokens_generated" not in st.session_state:
        st.session_state.total_tokens_generated = 0
    if "connection_status" not in st.session_state:
        st.session_state.connection_status = False

def auto_connect():
    """Tự động kết nối đến port 8000"""
    if not st.session_state.connection_status:
        with st.spinner("Đang tự động kết nối đến localhost:8000..."):
            st.session_state.vllm_client = vLLMClient("http://localhost:8000")
            st.session_state.models = st.session_state.vllm_client.get_models()
            if st.session_state.models:
                st.session_state.current_model = st.session_state.models[0]  # Chọn model đầu tiên
                st.session_state.connection_status = True
                st.success(f"Kết nối thành công! Sử dụng model: {st.session_state.current_model}")
            else:
                st.error("Không thể kết nối hoặc không tìm thấy models")

def main():
    init_session_state()
    
    st.title("🤖 vLLM Chatbot")
    st.markdown("---")
    
    # Tự động kết nối khi khởi động
    auto_connect()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Cấu hình")
        
        # Thống kê
        st.subheader("📊 Thống kê")
        
        # Đếm số turn (mỗi cặp user-assistant = 1 turn)
        user_messages = [msg for msg in st.session_state.messages if msg["role"] == "user"]
        assistant_messages = [msg for msg in st.session_state.messages if msg["role"] == "assistant"]
        turns = min(len(user_messages), len(assistant_messages))
        
        st.metric("Số turn", turns)
        st.metric("Tổng tin nhắn", len(st.session_state.messages))
        st.metric("Tokens đã tạo", st.session_state.total_tokens_generated)
        
        if st.session_state.current_model:
            st.metric("Model hiện tại", st.session_state.current_model)
        
        st.markdown("---")
        
        # Kết nối thủ công nếu cần
        st.subheader("Server Configuration")
        server_url = st.text_input(
            "vLLM Server URL",
            value="http://localhost:8000",
            help="URL của vLLM server"
        )
        
        if st.button("🔄 Kết nối lại"):
            with st.spinner("Đang kết nối..."):
                st.session_state.vllm_client = vLLMClient(server_url)
                st.session_state.models = st.session_state.vllm_client.get_models()
                if st.session_state.models:
                    st.session_state.current_model = st.session_state.models[0]
                    st.session_state.connection_status = True
                    st.success(f"Kết nối thành công! Model: {st.session_state.current_model}")
                else:
                    st.session_state.connection_status = False
                    st.error("Không thể kết nối")
        
        # Chọn model nếu có nhiều model
        if len(st.session_state.models) > 1:
            st.session_state.current_model = st.selectbox(
                "Chọn Model",
                st.session_state.models,
                index=st.session_state.models.index(st.session_state.current_model) if st.session_state.current_model in st.session_state.models else 0
            )
        
        st.markdown("---")
        
        # Cấu hình generation
        st.subheader("Generation Settings")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        max_tokens = st.slider("Max Tokens", 100, 4000, 1000, 100)
        top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.1)
        frequency_penalty = st.slider("Frequency Penalty", -2.0, 2.0, 0.0, 0.1)
        presence_penalty = st.slider("Presence Penalty", -2.0, 2.0, 0.0, 0.1)
        
        use_streaming = st.checkbox("Streaming Response", value=True)
        
        st.markdown("---")
        
        # System prompt
        st.subheader("System Prompt")
        system_prompt = st.text_area(
            "System Prompt",
            value="Bạn là một AI assistant hữu ích, thông minh và thân thiện. Hãy trả lời câu hỏi một cách chi tiết và chính xác.",
            height=100
        )
        
        # Clear chat
        if st.button("🗑️ Xóa lịch sử chat"):
            st.session_state.messages = []
            st.session_state.total_tokens_generated = 0
            st.rerun()
        
    
    # Main chat interface
    # Hiển thị messages
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.write(message["content"])
    
    # Input cho tin nhắn mới
    user_input = st.chat_input("Nhập tin nhắn của bạn...")
    
    if user_input and st.session_state.connection_status and st.session_state.current_model:
        # Thêm tin nhắn user vào history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Hiển thị tin nhắn user
        with st.chat_message("user"):
            st.write(user_input)
        
        # Chuẩn bị messages cho API
        api_messages = []
        if system_prompt.strip():
            api_messages.append({"role": "system", "content": system_prompt})
        
        api_messages.extend(st.session_state.messages)
        
        # Generate response
        with st.chat_message("assistant"):
            if use_streaming:
                # Streaming response
                response_placeholder = st.empty()
                full_response = ""
                
                for chunk in st.session_state.vllm_client.stream_chat_completion(
                    api_messages,
                    st.session_state.current_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty
                ):
                    full_response += chunk
                    response_placeholder.write(full_response + "▌")
                
                response_placeholder.write(full_response)
                response = full_response
                # Đếm token cho streaming response
                token_count = st.session_state.vllm_client.count_tokens(full_response)
            else:
                # Non-streaming response
                with st.spinner("Đang tạo phản hồi..."):
                    response, token_count = st.session_state.vllm_client.chat_completion(
                        api_messages,
                        st.session_state.current_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty
                    )
                st.write(response)
        
        # Thêm response vào history và cập nhật token count
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.total_tokens_generated += token_count
        st.rerun()
    
    elif user_input and not st.session_state.connection_status:
        st.error("Chưa kết nối đến server! Vui lòng kiểm tra kết nối.")

if __name__ == "__main__":
    main()