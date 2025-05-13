import os
import numpy as np
import pickle
import tenseal as ts
from tqdm import tqdm
import logging
from joblib import Parallel, delayed
import pandas as pd
import time
import sys
from deepface import DeepFace
import gc

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class BiometricEncryption:
    def __init__(self, poly_modulus_degree=16384, scale=2**40):
        """Initialize CKKS encryption parameters"""
        try:
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=poly_modulus_degree,
                coeff_mod_bit_sizes=[60, 40, 40, 40, 60]
            )
            self.context.global_scale = scale
            self.context.generate_galois_keys()
            self.context.generate_relin_keys()
            self.slot_count = poly_modulus_degree // 2
            logger.info("Encryption context initialized successfully")
            logger.info(f"Available slots for packing: {self.slot_count}")
        except Exception as e:
            logger.error(f"Error initializing encryption context: {str(e)}")
            raise

    def encrypt_vector(self, vector, packing_size=None):
        try:
            if packing_size is None:
                packing_size = min(len(vector), self.slot_count // 2)
            if len(vector) < packing_size:
                padded_vector = np.pad(vector, (0, packing_size - len(vector)))
            else:
                padded_vector = vector[:packing_size]
            return ts.ckks_vector(self.context, padded_vector.tolist())
        except Exception as e:
            logger.error(f"Error encrypting vector: {str(e)}")
            raise

def compute_cleartext_scores(test_embeddings, train_embeddings):
    """Compute similarity scores over cleartext data"""
    try:
        logger.info("Computing cleartext similarity scores...")
        start_time = time.time()
        
        n_test = len(test_embeddings)
        n_train = len(train_embeddings)
        scores = np.zeros((n_test, n_train))
        
        for i in tqdm(range(n_test), desc="Computing cleartext scores"):
            scores[i] = np.sum((train_embeddings - test_embeddings[i])**2, axis=1)
        
        runtime = time.time() - start_time
        logger.info(f"Cleartext computation time: {runtime:.2f} seconds")
        return scores
        
    except Exception as e:
        logger.error(f"Error in cleartext computation: {str(e)}")
        raise

def compute_encrypted_scores(test_embeddings, train_embeddings, crypto, batch_size=10):
    """Compute similarity scores over encrypted data"""
    try:
        logger.info("Computing encrypted similarity scores...")
        start_time = time.time()
        
        encrypted_scores = []
        for i in tqdm(range(0, len(test_embeddings), batch_size), desc="Processing batches"):
            batch = test_embeddings[i:i + batch_size]
            batch_scores = []
            
            for test_vec in batch:
                row_scores = []
                encrypted_test = crypto.encrypt_vector(test_vec)
                
                for train_vec in train_embeddings:
                    encrypted_train = crypto.encrypt_vector(train_vec)
                    diff = encrypted_test - encrypted_train
                    score = (diff * diff).sum()
                    row_scores.append(score)
                
                batch_scores.extend(row_scores)
                del encrypted_test
                gc.collect()
            
            encrypted_scores.extend(batch_scores)
        
        runtime = time.time() - start_time
        logger.info(f"Encrypted computation time: {runtime:.2f} seconds")
        return encrypted_scores
        
    except Exception as e:
        logger.error(f"Error in encrypted computation: {str(e)}")
        raise

def main():
    try:
        logger.info("Starting biometric identification process...")
        
        # Initialize encryption
        logger.info("Initializing encryption...")
        crypto = BiometricEncryption()
        
        # Load embeddings
        embeddings_file = "embeddings_Facenet.pkl"
        logger.info("Loading embeddings...")
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
        
        # Process embeddings
        numeric_embeddings = []
        for emb in embeddings.values():
            if isinstance(emb, list) and len(emb) > 0 and 'embedding' in emb[0]:
                numeric_embeddings.append(emb[0]['embedding'])
        numeric_embeddings = np.array(numeric_embeddings)
        
        # Split data
        train_ratio = 0.8
        split_idx = int(len(numeric_embeddings) * train_ratio)
        train_embeddings = numeric_embeddings[:split_idx]
        test_embeddings = numeric_embeddings[split_idx:]
        
        logger.info(f"Loaded {len(train_embeddings)} training and {len(test_embeddings)} test embeddings")
        
        # Compute cleartext scores
        cleartext_scores = compute_cleartext_scores(test_embeddings, train_embeddings)
        
        # Compute encrypted scores
        encrypted_scores = compute_encrypted_scores(test_embeddings, train_embeddings, crypto)
        
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
