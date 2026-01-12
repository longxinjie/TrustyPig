# --- Flask & CORS Setup ---
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS # to allow requests to a different domain

# --- Utility Libraries ---
from datetime import datetime, timezone
import os
from dotenv import load_dotenv 
load_dotenv() # loads environment variables from .env into Python environment

# --- ML & Data Handling ---
import joblib # a python library used mainly for saving and loading training ML models
import numpy as np

# --- Initialize Firebase (Admin SDK + Firestore Client) ---
import firebase_admin
from firebase_admin import credentials, firestore, auth

# The Firebase Admin SDK is initialised using a service account key file which is a secure JSON file downloaded from the Firebase project.
cred_path = os.getenv("FIREBASE_CREDENTIALS")
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred) # create a Firebase app instance on this Flask server

db = firestore.client() # access to Firestore

# --- Stripe SDK ---
import stripe

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
STRIPE_PUBLIC_KEY = os.getenv("STRIPE_PUBLIC_KEY")

# --- Load Model Artifacts ---
MODEL_PATH = os.getenv("MODEL_PATH", "model/fraud_model.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

model = joblib.load(MODEL_PATH)

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app) 

# --- Firebase Config ---
def get_firebase_config():
    return {
        "apiKey": os.getenv("FIREBASE_API_KEY"),
        "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
        "projectId": os.getenv("FIREBASE_PROJECT_ID"),
        "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
        "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
        "appId": os.getenv("FIREBASE_APP_ID"),
    }

# --- MAIN PAGE --- 
@app.route('/')
def index():
    return render_template('index.html')

# --- Login Page ---
@app.route('/login')
def login():
    return render_template("login.html", firebase_config=get_firebase_config())

# --- Registration Page ---
@app.route("/register")
def register():
    return render_template("register.html", firebase_config=get_firebase_config())

# --- OTP Page ---
@app.route('/verify-otp')
def verify_otp():
    return render_template("verify-otp.html", firebase_config=get_firebase_config())

# --- API Endpoint to create customer on Stripe ---
@app.route("/create-stripe-customer", methods=["POST"])
def create_stripe_customer():
    data = request.get_json()

    name = data.get("name")
    phone = data.get("phone")
    uid = data.get("uid")
    email = data.get("email") 

    if not all([name, phone, uid]):
        return jsonify({"error": "Missing required fields"}), 400

    customer = stripe.Customer.create(
        name=name,
        phone=phone,
        email=email, 
        metadata={"firebase_uid": uid}
    )

    return jsonify({"customer_id": customer.id})

# --- Success Registration Page ---
@app.route('/success-registration')
def success_registration():
    return render_template("success-registration.html")

# --- Add Card Page ---
@app.route("/add-card", methods=["GET", "POST"])
def add_card_page():
    if request.method == "GET":
        return render_template("add-card.html", firebase_config=get_firebase_config(), stripe_public_key=STRIPE_PUBLIC_KEY)

    if request.method == "POST":
        data = request.get_json()
        uid = data.get("uid")
        token = data.get("token")

        doc = db.collection("users").document(uid).get()
        if not doc.exists:
            return jsonify({"success": False, "error": "User not found"}), 404

        customer_id = doc.to_dict().get("stripeCustomerId")

        try:
            stripe.Customer.create_source(
                customer_id,
                source=token
            )
            return jsonify({"success": True}), 200
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/get-linked-card", methods=["POST"])
def get_linked_card():
    id_token = request.json.get("idToken")

    if not id_token:
        return jsonify({"error": "Missing ID token"}), 401

    try:
        # 1. Verify Firebase ID token and extract UID
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token["uid"]

        # 2. Look up Firestore 'users' collection
        doc = db.collection("users").document(uid).get()
        if not doc.exists:
            return jsonify({"error": "User not found in Firestore"}), 404

        customer_id = doc.to_dict().get("stripeCustomerId")
        if not customer_id:
            return jsonify({"error": "No Stripe customer ID stored in Firestore"}), 404

        # 3. Fetch Stripe cards using stored customer ID
        cards = stripe.PaymentMethod.list(
            customer=customer_id,
            type="card"
        )

        if not cards.data:
            return jsonify({"error": "No linked card found"}), 404

        card = cards.data[0].card
        return jsonify({
            "brand": card["brand"],
            "last4": card["last4"],
            "exp_month": card["exp_month"],
            "exp_year": card["exp_year"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@app.route('/home')
def home():
    return render_template('homepage.html')

@app.route("/friends")
def friends():
    return render_template("friends.html")

@app.route("/add-friends")
def add_friends():
    return render_template("add-friends.html")

@app.route('/wallet')
def wallet():
    return render_template('wallet.html')

@app.route('/piggypay')
def piggypay():
    return render_template('piggypay.html')

@app.route('/success_piggypay')
def success_piggypay():
    return render_template("success_piggypay.html")

@app.route('/topup')
def top_up():
    return render_template("topup.html")

@app.route("/success_topup")
def success_topup():
    return render_template("success_topup.html")

@app.route("/withdraw")
def withdraw():
    return render_template("withdraw.html")

@app.route("/success_withdraw")
def success_withdraw():
    return render_template("success_withdraw.html")

@app.route("/api/save-iban", methods=["POST"])
def save_iban():
    data = request.get_json()
    id_token = data.get("idToken")
    iban = data.get("iban")

    try:
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token["uid"]

        db.collection("users").document(uid).set({"iban": iban}, merge=True)

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    features = np.array([[
        data['wallet_ratio'], data['hour_of_day'], data['amount'],
        data['receiver_freq'], data['sender_freq'], data['is_merchant'],
        data['type_CASH_IN'], data['type_CASH_OUT'], data['type_DEBIT'],
        data['type_PAYMENT'], data['type_TRANSFER']
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return jsonify({
        'prediction': int(prediction),
        'probability': float(probability)
    })

# Add this route to app.py
@app.route("/api/transaction", methods=["POST"])
def unified_transaction():
    data = request.get_json()
    tx_type = data.get("txType")  # 'CASH_IN', 'CASH_OUT', 'TRANSFER'
    id_token = data.get("idToken")
    amount = float(data.get("amount", 0))
    contact = data.get("contact")  # only for TRANSFER

    if not id_token or not tx_type or amount <= 0:
        return jsonify({"success": False, "message": "Missing or invalid fields"}), 400

    try:
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token["uid"]
        user_ref = db.collection("users").document(uid)
        user_doc = user_ref.get()
        if not user_doc.exists:
            return jsonify({"error": "User not found"}), 404

        user_data = user_doc.to_dict()
        phone = user_data.get("phone", "unknown")
        balance = float(user_data.get("balance", 0))
        now = datetime.now(timezone.utc)
        hour = now.hour

        txns = list(user_ref.collection("transactions").stream())
        txns_data = [t.to_dict() for t in txns]
        sender_freq = len(txns_data)
        receiver_freq = sum(1 for t in txns_data if t.get("type") == tx_type)
        wallet_ratio = amount / balance if balance > 0 else 0.5
        is_merchant = 0

        all_types = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
        type_flags = {f"type_{t}": int(t == tx_type) for t in all_types}

        features = {
            "wallet_ratio": wallet_ratio,
            "hour_of_day": hour,
            "amount": amount,
            "receiver_freq": receiver_freq,
            "sender_freq": sender_freq,
            "is_merchant": is_merchant,
            **type_flags
        }

        prediction = model.predict([list(features.values())])[0]
        fraud_score = model.predict_proba([list(features.values())])[0][1]
        is_fraud = bool(prediction)
        is_flagged = bool(fraud_score >= THRESHOLD)

        txn_data = {
            "type": tx_type,
            "amount": amount,
            "timestamp": now,
            "fraud": is_fraud,
            "fraud_score": float(fraud_score),
            "verified": not is_fraud,
            "prediction": int(prediction),
            "model_version": MODEL_VERSION,
            "label": "pending" if is_fraud else "legit",
            "wallet_ratio": wallet_ratio,
            "hour_of_day": hour,
            "sender_freq": sender_freq,
            "receiver_freq": receiver_freq,
            "is_merchant": is_merchant,
            "isFlagged": is_flagged
        }

        # TRANSFER LOGIC
        if tx_type == "TRANSFER":
            if not contact:
                return jsonify({"success": False, "message": "Missing contact for PiggyPay"}), 400
            contact = contact if contact.startswith("+65") else f"+65{contact}"
            if contact == phone:
                return jsonify({"success": False, "message": "Cannot send to self"}), 400

            recipient_docs = list(db.collection("users").where("phone", "==", contact).limit(1).stream())
            if not recipient_docs:
                return jsonify({"success": False, "message": "Recipient not found"}), 404

            recipient_doc = recipient_docs[0]
            recipient_uid = recipient_doc.id

            if balance < amount:
                return jsonify({"success": False, "message": "Insufficient balance"}), 400

            if is_fraud:
                # Log but do NOT credit recipient yet
                user_ref.update({"has_fraud_alert": True})
                user_ref.collection("transactions").add({
                    **txn_data,
                    "direction": "out",
                    "type": f"TRANSFER Sent to {contact}",
                    "counterparty": contact,
                    "recipient_uid": recipient_uid,
                    "verified": False,
                    "flag_history": True
                })
                return jsonify({
                    "success": True,
                    "flagged": True,
                    "fraud": True,
                    "fraud_score": float(fraud_score),
                    "contact": contact,
                    "recipient_uid": recipient_uid
                }), 200

            # If NOT fraud: proceed with balance changes
            user_ref.update({"balance": firestore.Increment(-amount)})
            user_ref.collection("transactions").add({
                **txn_data,
                "direction": "out",
                "type": f"TRANSFER Sent to {contact}",
                "counterparty": contact,
                "verified": True
            })

            db.collection("users").document(recipient_uid).update({"balance": firestore.Increment(amount)})
            db.collection("users").document(recipient_uid).collection("transactions").add({
                **txn_data,
                "amount": amount,
                "direction": "in",
                "type": f"TRANSFER Received from {phone}",
                "counterparty": phone,
                "verified": True
            })

            return jsonify({
                "success": True,
                "fraud": False,
                "fraud_score": float(fraud_score),
                "contact": contact
            }), 200

        # CASH_IN
        elif tx_type == "CASH_IN":
            user_ref.collection("transactions").add(txn_data)

            if is_fraud:
                user_ref.update({"has_fraud_alert": True})
                return jsonify({
                    "success": True,
                    "flagged": True,
                    "fraud": True,
                    "fraud_score": float(fraud_score),
                    "contact": None
                }), 200

            user_ref.update({"balance": firestore.Increment(amount)})

            return jsonify({
                "success": True,
                "fraud": False,
                "fraud_score": float(fraud_score),
                "contact": None
            }), 200

        # CASH_OUT
        elif tx_type == "CASH_OUT":

            if balance < amount:
                return jsonify({"success": False, "message": "Insufficient balance"}), 400

            user_ref.collection("transactions").add(txn_data)

            if is_fraud:
                user_ref.update({"has_fraud_alert": True})
                return jsonify({
                    "success": True,
                    "flagged": True,
                    "fraud": True,
                    "fraud_score": float(fraud_score),
                    "contact": None
                }), 200

            user_ref.update({"balance": firestore.Increment(-amount)})

            return jsonify({
                "success": True,
                "fraud": False,
                "fraud_score": float(fraud_score),
                "contact": None
            }), 200

    except Exception as e:
        print("/api/transaction error:", e)
        return jsonify({"success": False, "message": str(e)}), 500

# OTP Verification Finalization Endpoint
@app.route("/api/verify-transaction", methods=["POST"])
def verify_transaction():
    data = request.get_json()
    id_token = data.get("idToken")

    try:
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token["uid"]

        user_ref = db.collection("users").document(uid)
        txns_ref = user_ref.collection("transactions")
        txns = txns_ref.where("verified", "==", False).stream()

        for txn in txns:
            txn_data = txn.to_dict()
            tx_type = txn_data.get("type")
            amount = txn_data.get("amount")
            direction = txn_data.get("direction", "out")

            txns_ref.document(txn.id).update({"verified": True})
            txns_ref.document(txn.id).update({"fraud": False})
            txns_ref.document(txn.id).update({
                "label": "legit",
                "resolved_at": datetime.now(timezone.utc).isoformat()
            })

            if tx_type.startswith("TRANSFER") and direction == "in":
                user_ref.update({"balance": firestore.Increment(amount)})
            elif tx_type == "CASH_IN":
                user_ref.update({"balance": firestore.Increment(amount)})
            elif tx_type == "CASH_OUT":
                user_ref.update({"balance": firestore.Increment(-amount)})

        user_ref.update({"has_fraud_alert": False})
        user_ref.update({"fraud": False})

        return jsonify({"success": True})

    except Exception as e:
        print("verify_transaction error:", e)
        return jsonify({"success": False, "message": str(e)}), 500
    
@app.route("/api/user", methods=["POST"])
def get_user_info():
    data = request.get_json()
    id_token = data.get("idToken")

    try:
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token["uid"]

        user_doc = db.collection("users").document(uid).get()
        if not user_doc.exists:
            return jsonify({"error": "User not found"}), 404

        user_data = user_doc.to_dict()
        return jsonify({
            "iban": user_data.get("iban", ""),
            "connectedAccountId": user_data.get("connectedAccountId", ""),
            "balance": user_data.get("balance", 0)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@app.route("/fraudsight")
def fraudsight():
    return render_template("fraudsight.html")

@app.route("/api/fraudsight-data", methods=["POST"])
def fraudsight_data():
    data = request.get_json()
    id_token = data.get("idToken")

    try:
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token["uid"]

        txns_ref = db.collection("users").document(uid).collection("transactions")
        txns = txns_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(100).stream()

        cleaned = []
        for doc in txns:
            d = doc.to_dict()
            if "timestamp" not in d:
                continue
            cleaned.append({
                "date": d["timestamp"].strftime("%Y-%m-%d") if hasattr(d["timestamp"], 'strftime') else d["timestamp"].to_datetime().strftime("%Y-%m-%d"),
                "amount": float(d.get("amount", 0)),
                "wallet_ratio": float(d.get("wallet_ratio", 0)),
                "sender_freq": int(d.get("sender_freq", 0)),
                "receiver_freq": int(d.get("receiver_freq", 0)),
                "type": d.get("type", "UNKNOWN")
            })

        return jsonify(cleaned)

    except Exception as e:
        print("/api/fraudsight-data error:", e)
        return jsonify({"error": str(e)}), 500
    
def log_prediction(txn_id, uid, raw, proba, prediction, status):
    """Log transaction decisions for future retraining."""
    db.collection("fraud_predictions").document(txn_id).set({
        "txn_id": txn_id,
        "uid": uid,
        "ts": datetime.now(timezone.utc).isoformat(),
        "raw": raw,
        "proba": float(proba),
        "prediction": int(prediction),
        "model_version": MODEL_VERSION,
        "label": status   # "pending", "legit", or "fraud"
    }, merge=True)


if __name__ == '__main__':
    app.run(debug=True)