import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore, auth
import os
from dotenv import load_dotenv
load_dotenv()

def export_data():
    
    cred_path = os.getenv("FIREBASE_CREDENTIALS")
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    all_txns = []
    users = db.collection("users").stream()
    for u in users:
        txns = db.collection("users").document(u.id).collection("transactions").stream()
        for t in txns:
            d = t.to_dict()
            if d.get("label") in ["fraud", "legit"]:
                all_txns.append(d)

    df = pd.DataFrame(all_txns)
    df.to_csv("data/app_transactions.csv", index=False)  
    print(f"Exported {len(df)} rows to data/app_transactions.csv")

if __name__ == "__main__":
    export_data()
