"""
Ed25519 key management and signature utilities for RSI patches.
"""
import os
import json
from nacl.signing import SigningKey, VerifyKey
from nacl.encoding import HexEncoder
from nacl.exceptions import BadSignatureError

KEYS_DIR = os.path.join(os.path.dirname(__file__), '../../data/keys')


def ensure_keys_exist(key_name: str = 'rsi_signer'):
    os.makedirs(KEYS_DIR, exist_ok=True)
    priv_path = os.path.join(KEYS_DIR, f'{key_name}_private.key')
    pub_path = os.path.join(KEYS_DIR, f'{key_name}_public.key')
    if not os.path.exists(priv_path):
        sk = SigningKey.generate()
        with open(priv_path, 'wb') as f:
            f.write(sk.encode())
        with open(pub_path, 'wb') as f:
            f.write(sk.verify_key.encode())
    return priv_path, pub_path

def generate_keypair(key_name: str = 'rsi_signer'):
    """
    Generate and store an Ed25519 keypair in data/keys/.
    """
    key_dir = os.path.join(os.path.dirname(__file__), '../../data/keys')
    os.makedirs(key_dir, exist_ok=True)
    sk = SigningKey.generate()
    with open(os.path.join(key_dir, f'{key_name}.priv'), 'wb') as f:
        f.write(sk.encode())
    with open(os.path.join(key_dir, f'{key_name}.pub'), 'wb') as f:
        f.write(sk.verify_key.encode())
    return True

def load_signing_key(key_name: str = 'rsi_signer') -> SigningKey:
    priv_path, _ = ensure_keys_exist(key_name)
    with open(priv_path, 'rb') as f:
        return SigningKey(f.read())

def load_verify_key(key_name: str = 'rsi_signer') -> VerifyKey:
    _, pub_path = ensure_keys_exist(key_name)
    with open(pub_path, 'rb') as f:
        return VerifyKey(f.read())

def sign_patch(patch_data: dict, key_name: str = 'rsi_signer') -> str:
    """
    Sign the canonical JSON serialization of patch_data (sorted keys, utf-8 encoded).
    """
    if not isinstance(patch_data, dict):
        raise TypeError("patch_data must be a dictionary")
    patch_bytes = json.dumps(patch_data, sort_keys=True).encode("utf-8")
    sk = load_signing_key(key_name)
    sig = sk.sign(patch_bytes).signature
    return sig.hex()

def verify_patch_signature(patch_data: dict, signature_hex: str, key_name: str = 'rsi_signer') -> bool:
    """
    Verify the signature against the canonical JSON serialization of patch_data (sorted keys, utf-8 encoded).
    """
    if not isinstance(patch_data, dict):
        raise TypeError("patch_data must be a dictionary")
    patch_bytes = json.dumps(patch_data, sort_keys=True).encode("utf-8")
    vk = load_verify_key(key_name)
    try:
        vk.verify(patch_bytes, bytes.fromhex(signature_hex))
        return True
    except BadSignatureError:
        return False
