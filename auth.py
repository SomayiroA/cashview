from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from typing import List

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Simple in-memory user store for demonstration
users_db = {
    "atm_ops": {"username": "atm_ops", "role": "ATM Operations Staff"},
    "branch_mgr": {"username": "branch_mgr", "role": "Branch Operations Manager"},
    "vault_mgr": {"username": "vault_mgr", "role": "Vault Manager"},
    "hoao": {"username": "hoao", "role": "Head Office Authorization Officer"},
}

def get_current_user(token: str = Depends(oauth2_scheme)):
    user = users_db.get(token)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
    return user

def role_required(allowed_roles: List[str]):
    def role_checker(user: dict = Depends(get_current_user)):
        if user["role"] not in allowed_roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Operation not permitted for your role")
        return user
    return role_checker
