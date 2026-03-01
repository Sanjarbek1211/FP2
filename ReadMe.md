## ⚽ Football Match Outcome Prediction
📌 Project Overview

Ushbu loyiha futbol o‘yinlari natijasini (multi-class classification) bashorat qilish uchun yaratilgan.
Model oxirgi mavsum statistikalariga asoslanib kelajakdagi o‘yin natijalarini prognoz qiladi.

🎯 Maqsad

O‘yin natijasini bashorat qilish

Model qarorini SHAP orqali tushuntirish

Data leakage’ni oldini olish

Vaqt bo‘yicha (season-based) realistik baholash

📊 Dataset

Featurelar:

home_last10_points

away_last10_points

home_last10_gd

away_last10_gd

h2h_last5_home_points

h2h_last5_away_points

season_start

home_team

away_team

Target:

0 → Class 0

1 → Class 1

2 → Class 2

(natija mapping loyihaga qarab aniqlanadi)

🔬 Data Splitting Strategy

Data vaqt bo‘yicha ajratildi:

Oxirgi 3 season → Test set

Oldingi seasonlar → Train set

Bu real forecasting sharoitini simulyatsiya qiladi.

🧠 Modellar

Sinovdan o‘tgan modelllar:

Logistic Regression

Random Forest

Random Forest (Feature Selection)

Random Forest (Hyperparameter Tuning)

Asosiy baholash metrikasi:

F1 Macro

📈 Natijalar (Realistic)
Model	Accuracy	F1 Macro
RF (Tree FS)	~0.50	~0.47
RF (Grid Tuning)	~0.50	~0.40

Leakage mavjud bo‘lgan baseline natijalar sun’iy yuqori bo‘lgan va chiqarib tashlangan.

🔍 Model Interpretation

SHAP yordamida:

Global feature importance tahlil qilindi

Local (bitta match) tushuntirish qilindi

home_last10_gd va away_last10_gd eng muhim feature ekanligi aniqlandi

season_start yuqori chiqishi temporal pattern mavjudligini ko‘rsatdi

⚠ Muhim Kuzatuvlar

Draw classni bashorat qilish qiyin

Season bo‘yicha pattern o‘zgarishi (concept drift) ehtimoli mavjud

Ordinal encoding teamlar uchun bias berishi mumkin

🚀 Keyingi Rivojlantirish

Rolling form featurelarini kengaytirish

Elo rating qo‘shish

OneHotEncoding sinab ko‘rish

Class imbalance muammosini yaxshilash

Time-series cross validation

🛠 Texnologiyalar

Python

Pandas

Scikit-learn

SHAP

Matplotlib

Tabulate

📌 Xulosa

Model real sharoitda 50% atrofida natija bermoqda.
Leakage bartaraf etilganidan so‘ng natijalar realistik ko‘rinishga keldi.

Loyiha model tushuntiriluvchanligini (explainability) ta’minlaydi va vaqt bo‘yicha generalizatsiyani hisobga oladi.
