README - Privacy-Preserving Biometric Identification

Project Overview
This project implements a privacy-preserving biometric identification system using Fully Homomorphic Encryption (FHE). The system is divided into three parts:
- Part A: Biometric identification using cleartext data with limited precision.
- Part B: Computation of similarity metrics over encrypted data.
- Part C: Full biometric identification over encrypted vectors.

Requirements
Make sure you have the following installed before running the project:
- Python 3.8+
- Required Python libraries:
  pip install -r requirements.txt
- DeepFace (https://github.com/serengil/deepface) for face embeddings
- Microsoft SEAL (https://github.com/microsoft/SEAL) or TenSEAL (https://github.com/OpenMined/TenSEAL) for homomorphic encryption

How to Run
1. Run Part A (Biometric Identification on Cleartext Data)
   python rec_model.py
   This script will:
   - Load face embeddings from the LFW dataset.
   - Compute similarity scores with different precision levels.
   - Save accuracy and performance results.

2. Run Part B (Homomorphic Similarity Computation)
   python enc_similarity.py
   This script will:
   - Encrypt sample vectors using CKKS FHE.
   - Compute similarity metrics over encrypted vectors.
   - Compare accuracy between cleartext and encrypted similarity scores.

3. Run Part C (Full Privacy-Preserving Biometric Identification)
   python biometric_identification.py
   This script will:
   - Load and encrypt embeddings.
   - Compute similarity scores using FHE.
   - Compare decrypted results with cleartext identification.
   - Save results in CSV files.

Output Files
- scores.csv: Cleartext similarity scores.
- scores_dec.csv: Decrypted homomorphic similarity scores.
- top10.csv: Top-10 identification results (cleartext).
- top10_dec.csv: Top-10 identification results (encrypted computation).
- results_report.txt: Accuracy and performance metrics.

Notes
- The scripts assume the LFW dataset is stored in ./datasets/lfw/lfw-deepfunneled/
- If needed, embeddings will be recalculated and saved as embeddings_Facenet.pkl
- Some scripts might take time due to encryption overhead.

If you have any issues running the project, let me know!

---
Authors: Amal Awawdi  & John Haddad 
