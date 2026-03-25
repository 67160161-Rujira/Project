# 🏠 แอปทำนายราคาบ้าน (House Price Prediction)

แอปพลิเคชัน Machine Learning สำหรับทำนายราคาบ้าน พัฒนาและแสดงผลผ่าน Streamlit

---

## 📌 ภาพรวมโปรเจค

โปรเจคนี้มีวัตถุประสงค์เพื่อทำนายราคาบ้านจากข้อมูลคุณสมบัติของอสังหาริมทรัพย์ เช่น จำนวนห้อง ขนาดพื้นที่ และตำแหน่งที่ตั้ง

โมเดลถูกพัฒนาโดยใช้ **Machine Learning Pipeline** ซึ่งรวมขั้นตอนการเตรียมข้อมูลและโมเดลไว้ในระบบเดียว

---

## 🚀 ความสามารถของระบบ

* 📊 ทำนายราคาบ้านแบบเรียลไทม์
* 🌏 รองรับหลายภูมิภาค
* 💰 แสดงราคาเป็น **AUD และ THB**
* 🤖 แสดงค่าความมั่นใจ (Confidence Score)
* 🎨 มี UI สวยงาม ใช้งานง่าย

---

## 🧠 รายละเอียดโมเดล

* อัลกอริทึม: Random Forest Regressor
* การเตรียมข้อมูล:

  * StandardScaler (ข้อมูลตัวเลข)
  * OneHotEncoder (ข้อมูลประเภท)
* การประเมินโมเดล:

  * R² ≈ **0.82**
  * Cross-validation ≈ **0.76**

---

## 📂 โครงสร้างโปรเจค

```id="x8c2p1"
project/
│
├── app.py                 # แอป Streamlit
├── house_model.pkl       # โมเดลที่เทรนแล้ว
├── requirements.txt      # ไลบรารีที่ใช้
└── README.md             # เอกสารโปรเจค
```

---

## ⚙️ วิธีติดตั้ง

Clone โปรเจค:

```bash id="u2m91k"
git clone https://github.com/your-username/house-price-app.git
cd house-price-app
```

ติดตั้งไลบรารี:

```bash id="p4f82a"
pip install -r requirements.txt
```

---

## ▶️ วิธีรันโปรแกรม

```bash id="z8l3kd"
streamlit run app.py
```

---

## 📊 ข้อมูลที่ใช้ทำนาย

* Rooms (จำนวนห้อง)
* Bedrooms (ห้องนอน)
* Bathrooms (ห้องน้ำ)
* Car Spaces (ที่จอดรถ)
* Land Size (ขนาดที่ดิน)
* Building Area (ขนาดอาคาร)
* Year Built (ปีที่สร้าง)
* Distance (ระยะทางจากเมือง)
* Region (ภูมิภาค)

---

## 📈 ผลลัพธ์ที่ได้

* ราคาบ้าน (AUD 💵)
* ราคาบ้าน (THB 🇹🇭)
* ค่า Confidence (%)

---

## ⚠️ หมายเหตุ

ราคาที่แสดงเป็นเพียงค่าประมาณจากโมเดล Machine Learning
อาจไม่ตรงกับราคาจริงในตลาด

---

## 👩‍💻 ผู้พัฒนา

รุจิรา แซ่เตีย
