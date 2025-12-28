# Dockerfile tối ưu hơn
FROM python:3.11.2-slim-bullseye

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Chỉ sao chép tệp requirements.txt trước
COPY requirements.txt .

# Cài đặt các thư viện. Lớp này sẽ được cache lại
# miễn là tệp requirements.txt không thay đổi.
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Bây giờ mới sao chép toàn bộ mã nguồn ứng dụng
COPY . .

# Đảm bảo app là package
RUN touch app/__init__.py