import uuid
from fastapi import APIRouter, HTTPException

from app.models.schemas import UserCreate, UserUpdate, UserInfo
from app.services.database import mysql as mysql_service

router = APIRouter()


@router.post("", response_model=UserInfo)
async def create_user(user: UserCreate):
    """사용자 생성"""
    # 이메일 중복 체크
    existing = mysql_service.get_user_by_email(user.email)
    if existing:
        raise HTTPException(status_code=400, detail="이미 존재하는 이메일입니다.")
    
    user_id = str(uuid.uuid4())
    
    success = mysql_service.create_user(
        user_id=user_id,
        username=user.username,
        email=user.email,
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="사용자 생성 실패")
    
    created = mysql_service.get_user_by_id(user_id)
    return UserInfo(
        user_id=created["id"],
        username=created["username"],
        email=created["email"],
        created_at=created.get("created_at"),
    )


@router.get("/{user_id}", response_model=UserInfo)
async def get_user(user_id: str):
    """사용자 조회"""
    user = mysql_service.get_user_by_id(user_id)
    
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    
    return UserInfo(
        user_id=user["id"],
        username=user["username"],
        email=user["email"],
        created_at=user.get("created_at"),
    )


@router.put("/{user_id}", response_model=UserInfo)
async def update_user(user_id: str, update: UserUpdate):
    """사용자 수정"""
    existing = mysql_service.get_user_by_id(user_id)
    if not existing:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    
    # 이메일 변경 시 중복 체크
    if update.email and update.email != existing["email"]:
        email_exists = mysql_service.get_user_by_email(update.email)
        if email_exists:
            raise HTTPException(status_code=400, detail="이미 존재하는 이메일입니다.")
    
    success = mysql_service.update_user(
        user_id=user_id,
        username=update.username,
        email=update.email,
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="사용자 수정 실패")
    
    updated = mysql_service.get_user_by_id(user_id)
    return UserInfo(
        user_id=updated["id"],
        username=updated["username"],
        email=updated["email"],
        created_at=updated.get("created_at"),
    )


@router.delete("/{user_id}")
async def delete_user(user_id: str):
    """사용자 삭제"""
    existing = mysql_service.get_user_by_id(user_id)
    if not existing:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    
    success = mysql_service.delete_user(user_id)
    if not success:
        raise HTTPException(status_code=500, detail="사용자 삭제 실패")
    
    return {"message": "사용자가 삭제되었습니다."}
