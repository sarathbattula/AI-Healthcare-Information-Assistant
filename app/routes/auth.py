from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.security import (
    hash_password, verify_password, create_access_token,
    get_current_user, log_audit
)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

class RegisterInput(BaseModel):
    username: str
    email: str
    password: str
    role: str = "user"

class UserOutput(BaseModel):
    id: int
    username: str
    email: str
    role: str

class TokenOutput(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOutput

@router.post("/register", response_model=UserOutput)
def register_user(payload: RegisterInput, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == payload.username).first():
        raise HTTPException(status_code=400, detail="Username already registered.")
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=400, detail="Email already registered.")

    hashed = hash_password(payload.password)
    new_user = User(
        username=payload.username,
        email=payload.email,
        hashed_password=hashed,
        role=payload.role if payload.role in ["user", "admin", "reviewer"] else "user"
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    log_audit(db, new_user.id, "USER_REGISTER", "users", str(new_user.id), f"User {new_user.username} registered.")
    return new_user

@router.post("/token", response_model=TokenOutput)
def login_user(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password.")
    if not user.is_active:
        raise HTTPException(status_code=400, detail="User account is inactive.")

    token = create_access_token(data={"sub": user.username, "role": user.role, "id": user.id})
    log_audit(db, user.id, "USER_LOGIN", "users", str(user.id), "Successful login.")
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "role": user.role
        }
    }

@router.get("/me", response_model=UserOutput)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user
