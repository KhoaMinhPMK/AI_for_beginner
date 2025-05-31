# Lộ Trình Học Trí Tuệ Nhân Tạo Cho Học Sinh Cấp 3

## 🎯 Mục Tiêu Tổng Quát
Giúp học sinh cấp 3 hiểu và áp dụng AI trong thực tế thông qua các project: Tạo sinh ngôn ngữ, API AI, RAG, nhận diện hình ảnh YOLO, nhận diện hành động.

---

## 📅 GIAI ĐOẠN 1: CƠ BẢN (4 tuần)

### Tuần 1-2: Giới thiệu AI
**Mục tiêu:** Hiểu AI là gì và ứng dụng trong đời sống

**Nội dung:**
- AI là gì? Các ví dụ: Siri, Google Translate, Netflix gợi ý phim
- Phân biệt AI, Machine Learning, Deep Learning
- Lịch sử phát triển AI đơn giản
- Đạo đức trong AI

**Thực hành:**
- Sử dụng ChatGPT, Google Bard
- Chơi game AI: Quick Draw, AI Experiments
- Thảo luận về tác động AI đến xã hội

### Tuần 3-4: Lập trình cơ bản với Python
**Mục tiêu:** Nắm vững Python cơ bản

**Nội dung:**
- Cài đặt Python, VS Code
- Biến, kiểu dữ liệu, vòng lặp, điều kiện
- List, dictionary, function
- Thư viện cơ bản: requests, json

**Thực hành:**
- Viết chương trình đơn giản: máy tính, game đoán số
- Crawl dữ liệu đơn giản từ web
- **Mini Project:** Chatbot đơn giản với if-else

---

## 📅 GIAI ĐOẠN 2: TRUNG CẤP (6 tuần)

### Tuần 5-6: Làm việc với API
**Mục tiêu:** Hiểu và sử dụng API AI

**Nội dung:**
- API là gì? REST API cơ bản
- Đăng ký và sử dụng OpenAI API
- Xử lý JSON response
- Error handling

**Thực hành:**
- Gọi OpenAI API để tạo text
- Tạo chatbot đơn giản với API
- **Project 1:** Ứng dụng tạo câu chuyện tự động

### Tuần 7-8: Xử lý dữ liệu và Machine Learning cơ bản
**Mục tiêu:** Hiểu cách máy học từ dữ liệu

**Nội dung:**
- Pandas cơ bản: đọc, xử lý dữ liệu
- Visualization với matplotlib
- Giới thiệu sklearn: regression, classification đơn giản

**Thực hành:**
- Phân tích dataset đơn giản (iris, titanic)
- Vẽ biểu đồ, tìm pattern trong data
- **Project 2:** Dự đoán giá nhà đơn giản

### Tuần 9-10: Giới thiệu Deep Learning
**Mục tiêu:** Hiểu neural network cơ bản

**Nội dung:**
- Neural network là gì? (ví dụ não người)
- Tensorflow/Keras cơ bản
- Train model đơn giản

**Thực hành:**
- Tạo model phân loại hình ảnh đơn giản
- **Project 3:** Nhận diện chữ viết tay (MNIST)

---

## 📅 GIAI ĐOẠN 3: NÂNG CAO (8 tuần)

### Tuần 11-12: Computer Vision với YOLO
**Mục tiêu:** Nhận diện object trong hình ảnh

**Nội dung:**
- Computer Vision là gì?
- YOLO algorithm overview (đơn giản)
- Sử dụng YOLOv8 pre-trained
- OpenCV cơ bản

**Thực hành:**
- Detect objects trong ảnh/video
- **Project 4:** Ứng dụng đếm người qua camera

### Tuần 13-14: Natural Language Processing
**Mục tiêu:** Xử lý ngôn ngữ tự nhiên

**Nội dung:**
- NLP là gì? Tokenization, embedding
- Sử dụng transformers library
- Text generation với pre-trained models

**Thực hành:**
- Text summarization
- Sentiment analysis
- **Project 5:** Chatbot thông minh với Transformer

### Tuần 15-16: RAG (Retrieval Augmented Generation)
**Mục tiêu:** Xây dựng hệ thống QA thông minh

**Nội dung:**
- RAG là gì? Tại sao cần RAG?
- Vector database (ChromaDB)
- Embedding và similarity search

**Thực hành:**
- Tạo knowledge base từ documents
- **Project 6:** Chatbot trả lời câu hỏi về sách giáo khoa

### Tuần 17-18: Action Recognition
**Mục tiêu:** Nhận diện hành động từ video

**Nội dung:**
- Action recognition concepts
- Sử dụng MediaPipe
- Time series analysis cơ bản

**Thực hành:**
- Detect pose từ webcam
- **Project 7:** Đếm số lần tập thể dục

---

## 📅 GIAI ĐOẠN 4: PROJECT TỔNG HỢP (4 tuần)

### Tuần 19-22: Final Project
**Mục tiêu:** Tích hợp tất cả kiến thức

**Các lựa chọn project:**

#### Option 1: Smart Classroom Assistant
- Nhận diện học sinh qua camera (YOLO)
- Chatbot hỗ trợ học tập (RAG + API)
- Phân tích hành vi học tập (Action Recognition)

#### Option 2: Personal AI Tutor
- Tạo câu hỏi tự động (Language Generation)
- Chấm bài qua ảnh (Computer Vision)
- Gợi ý học tập cá nhân hóa (RAG)

#### Option 3: Content Creator Assistant
- Tạo nội dung từ keywords (API + Generation)
- Phân tích hình ảnh để gắn tag (YOLO)
- Tương tác qua video call (Action Recognition)

---

## 🛠️ Công Cụ và Thư Viện Chính

### Development Environment
- **Python 3.8+**
- **VS Code** với extensions Python
- **Jupyter Notebook** cho experiments
- **Git** cho version control

### Libraries
```python
# Cơ bản
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt

# Machine Learning
import sklearn
import tensorflow as tf
from transformers import pipeline

# Computer Vision
import cv2
from ultralytics import YOLO
import mediapipe as mp

# NLP & RAG
from langchain import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer
```

### APIs và Services
- **OpenAI API** (GPT-4, DALL-E)
- **Hugging Face** (free models)
- **Google Colab** (free GPU)

---

## 📚 Tài Liệu Học Tập

### Sách tham khảo
1. "Python for Everybody" - Charles Severance
2. "Hands-On Machine Learning" - Aurélien Géron (phiên bản đơn giản)

### Online Resources
- **Kaggle Learn** (miễn phí, có certificate)
- **Coursera AI for Everyone** (audit miễn phí)
- **YouTube:** Sentdex, 3Blue1Brown
- **Documentation:** TensorFlow, OpenCV, Transformers

### Datasets cho Practice
- **Computer Vision:** COCO, ImageNet subset
- **NLP:** WikiText, Common Crawl
- **Action:** UCF-101, Kinetics (subset)

---

## 🎯 Phương Pháp Đánh Giá

### Đánh giá theo từng giai đoạn (40%)
- Quiz trắc nghiệm
- Coding exercises
- Mini projects

### Project cuối kỳ (40%)
- Chức năng hoàn chỉnh
- Code quality
- Presentation

### Tham gia lớp (20%)
- Thảo luận
- Peer review
- Collaboration

---

## 🚀 Lộ Trình Mở Rộng

### Sau khóa học, học sinh có thể:
1. **Tham gia competition:** Kaggle, Zindi
2. **Xây dựng portfolio:** GitHub projects
3. **Học sâu hơn:** Chuyên ngành AI tại đại học
4. **Khởi nghiệp:** AI startup ideas

### Kết nối cộng đồng
- **Local AI meetups**
- **Online forums:** Reddit r/MachineLearning
- **Social media:** Follow AI researchers, practitioners

---

*"The best way to learn AI is by doing. Let's build the future together!"* 🤖✨
