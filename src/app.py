import streamlit as st
import requests
import json
from typing import List, Dict, Generator
import time
import tiktoken

# Cấu hình trang
st.set_page_config(
    page_title="Ollama Chatbot",
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

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
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
        """Lấy danh sách models có sẵn từ Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", headers=self.headers)
            if response.status_code == 200:
                models = response.json()
                return [model["name"] for model in models.get("models", [])]
            return []
        except Exception as e:
            st.error(f"Không thể kết nối tới Ollama server: {e}")
            return []
    
    def chat_completion(self, messages: List[Dict], model: str, **kwargs) -> tuple:
        """Gửi yêu cầu chat completion tới Ollama và trả về (response, token_count)"""
        
        # Chuyển đổi messages thành format của Ollama
        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        payload = {
            "model": model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_predict": kwargs.get("max_tokens", 1000),
                "top_p": kwargs.get("top_p", 0.9),
                "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                "presence_penalty": kwargs.get("presence_penalty", 0.0),
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                headers=self.headers,
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["message"]["content"]
                token_count = self.count_tokens(content)
                return content, token_count
            else:
                error_msg = f"Lỗi API: {response.status_code} - {response.text}"
                return error_msg, 0
                
        except Exception as e:
            error_msg = f"Lỗi kết nối: {str(e)}"
            return error_msg, 0
    
    def stream_chat_completion(self, messages: List[Dict], model: str, **kwargs) -> Generator[str, None, None]:
        """Streaming chat completion với Ollama"""
        
        # Chuyển đổi messages thành format của Ollama
        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        payload = {
            "model": model,
            "messages": ollama_messages,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_predict": kwargs.get("max_tokens", 1000),
                "top_p": kwargs.get("top_p", 0.9),
                "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                "presence_penalty": kwargs.get("presence_penalty", 0.0),
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=120
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            json_data = json.loads(line.decode('utf-8'))
                            if not json_data.get("done", False):
                                content = json_data.get("message", {}).get("content", "")
                                if content:
                                    yield content
                            else:
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                yield f"Lỗi API: {response.status_code}"
                
        except Exception as e:
            yield f"Lỗi kết nối: {str(e)}"

    def pull_model(self, model_name: str) -> Generator[str, None, None]:
        """Pull model từ Ollama registry"""
        payload = {
            "name": model_name,
            "stream": True
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=300
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            json_data = json.loads(line.decode('utf-8'))
                            status = json_data.get("status", "")
                            if status:
                                yield status
                        except json.JSONDecodeError:
                            continue
            else:
                yield f"Lỗi khi pull model: {response.status_code}"
                
        except Exception as e:
            yield f"Lỗi kết nối: {str(e)}"

def init_session_state():
    """Khởi tạo session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ollama_client" not in st.session_state:
        st.session_state.ollama_client = None
    if "models" not in st.session_state:
        st.session_state.models = []
    if "current_model" not in st.session_state:
        st.session_state.current_model = None
    if "total_tokens_generated" not in st.session_state:
        st.session_state.total_tokens_generated = 0
    if "connection_status" not in st.session_state:
        st.session_state.connection_status = False

def auto_connect():
    """Tự động kết nối tới Ollama tại port 11434"""
    if not st.session_state.connection_status:
        with st.spinner("Đang kết nối tới Ollama localhost:11434..."):
            st.session_state.ollama_client = OllamaClient("http://localhost:11434")
            st.session_state.models = st.session_state.ollama_client.get_models()
            if st.session_state.models:
                st.session_state.current_model = st.session_state.models[0]  # Chọn model đầu tiên
                st.session_state.connection_status = True
                st.success(f"Kết nối thành công! Model hiện tại: {st.session_state.current_model}")
            else:
                st.warning("Kết nối thành công nhưng không tìm thấy model nào. Vui lòng pull model trước!")

def main():
    init_session_state()
    
    st.title("🤖 Ollama Chatbot")
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
        
        st.metric("Số lượt hội thoại", turns)
        st.metric("Tổng tin nhắn", len(st.session_state.messages))
        st.metric("Token đã tạo", st.session_state.total_tokens_generated)
        
        if st.session_state.current_model:
            st.metric("Model hiện tại", st.session_state.current_model)
        
        st.markdown("---")
        
        # Cấu hình server
        st.subheader("🔧 Cấu hình Server")
        server_url = st.text_input(
            "Ollama Server URL",
            value="http://localhost:11434",
            help="URL của Ollama server"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Kết nối lại"):
                with st.spinner("Đang kết nối..."):
                    st.session_state.ollama_client = OllamaClient(server_url)
                    st.session_state.models = st.session_state.ollama_client.get_models()
                    if st.session_state.models:
                        st.session_state.current_model = st.session_state.models[0]
                        st.session_state.connection_status = True
                        st.success(f"Kết nối thành công! Model: {st.session_state.current_model}")
                    else:
                        st.session_state.connection_status = False
                        st.error("Kết nối thất bại hoặc không có model")
        
        with col2:
            if st.button("🔄 Refresh Models"):
                if st.session_state.ollama_client:
                    st.session_state.models = st.session_state.ollama_client.get_models()
                    if st.session_state.models:
                        st.success(f"Đã tải {len(st.session_state.models)} models")
                    else:
                        st.warning("Không tìm thấy model nào")
        
        # Pull model mới
        st.subheader("📥 Pull Model")
        model_to_pull = st.text_input("Tên model cần pull", placeholder="ví dụ: llama3.2:latest")
        if st.button("Pull Model") and model_to_pull:
            with st.spinner("Đang pull model..."):
                status_placeholder = st.empty()
                for status in st.session_state.ollama_client.pull_model(model_to_pull):
                    status_placeholder.text(f"Status: {status}")
                st.success("Pull model hoàn tất!")
                # Refresh models list
                st.session_state.models = st.session_state.ollama_client.get_models()
        
        # Chọn model
        if st.session_state.models:
            st.session_state.current_model = st.selectbox(
                "Chọn Model",
                st.session_state.models,
                index=st.session_state.models.index(st.session_state.current_model) if st.session_state.current_model in st.session_state.models else 0
            )
        
        st.markdown("---")
        
        # Cấu hình generation
        st.subheader("🎛️ Cài đặt Generation")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        max_tokens = st.slider("Max Tokens", 100, 4000, 1000, 100)
        top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.1)
        frequency_penalty = st.slider("Frequency Penalty", -2.0, 2.0, 0.0, 0.1)
        presence_penalty = st.slider("Presence Penalty", -2.0, 2.0, 0.0, 0.1)
        
        use_streaming = st.checkbox("Streaming Response", value=True)
        
        st.markdown("---")
        
        # System prompt
        st.subheader("💬 System Prompt")
        system_prompt = st.text_area(
            "System Prompt",
            value="Bạn là một AI assistant thông minh, hữu ích và thân thiện. Hãy trả lời một cách chi tiết và chính xác.",
            height=100
        )
        
        # Clear chat
        if st.button("🗑️ Xóa lịch sử chat"):
            st.session_state.messages = []
            st.session_state.total_tokens_generated = 0
            st.rerun()
        
        # Model info
        if st.session_state.current_model:
            st.markdown("---")
            st.subheader("ℹ️ Model Info")
            st.text(f"Model: {st.session_state.current_model}")
            st.text(f"Server: {server_url}")
    
    # Main chat interface
    # Hiển thị messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.write(message["content"])
    
    # Input cho tin nhắn mới
    user_input = st.chat_input("Nhập tin nhắn...")
    
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
                
                for chunk in st.session_state.ollama_client.stream_chat_completion(
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
                token_count = st.session_state.ollama_client.count_tokens(full_response)
            else:
                # Non-streaming response
                with st.spinner("Đang phản hồi..."):
                    response, token_count = st.session_state.ollama_client.chat_completion(
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
        st.error("Chưa kết nối tới Ollama Server! Vui lòng kiểm tra kết nối.")
    
    elif user_input and not st.session_state.current_model:
        st.error("Chưa chọn model! Vui lòng pull model hoặc chọn model có sẵn.")

if __name__ == "__main__":
    main()