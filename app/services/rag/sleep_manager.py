# -*- coding: utf-8 -*-
"""
vLLM Sleep Mode Manager

마지막 LLM 요청으로부터 IDLE_TIMEOUT 이상 경과 시 자동으로 슬립 모드 진입
새 요청 시 자동으로 깨우기
"""
import asyncio
import time
from datetime import datetime
from enum import Enum
from typing import Optional
import httpx

from app.core.structured_logger import get_logger

logger = get_logger(__name__)


class ModelState(str, Enum):
    """모델 상태"""
    ACTIVE = "ACTIVE"           # 정상 서빙 중
    SLEEPING = "SLEEPING"       # Level 1 슬립 (CPU RAM에 가중치)
    WAKING = "WAKING"           # 깨어나는 중


class VLLMSleepManager:
    """
    vLLM 슬립 모드 관리자

    Features:
    - 마지막 요청 시간 추적
    - Idle timeout 시 자동 슬립 (Level 1)
    - 새 요청 시 자동 wake
    - 백그라운드 태스크로 주기적 체크
    """

    def __init__(
        self,
        vllm_dev_url: str,
        idle_timeout: int = 300,  # 5분 (초 단위)
        check_interval: int = 60,  # 1분마다 체크
        enabled: bool = True
    ):
        """
        Args:
            vllm_dev_url: vLLM 개발 모드 엔드포인트 URL (예: http://localhost:8000)
            idle_timeout: 슬립 진입까지 대기 시간 (초)
            check_interval: 상태 체크 주기 (초)
            enabled: 슬립 모드 활성화 여부
        """
        self.vllm_dev_url = vllm_dev_url.rstrip('/')
        self.idle_timeout = idle_timeout
        self.check_interval = check_interval
        self.enabled = enabled

        # 상태 관리
        self._last_request_time: Optional[float] = None
        self._state: ModelState = ModelState.ACTIVE
        self._background_task: Optional[asyncio.Task] = None

        logger.info(
            f"VLLMSleepManager initialized: "
            f"enabled={enabled}, idle_timeout={idle_timeout}s, "
            f"check_interval={check_interval}s, url={vllm_dev_url}"
        )

    def update_last_request_time(self):
        """마지막 요청 시간 업데이트 (LLM 호출 시마다 호출)"""
        self._last_request_time = time.time()
        if self.enabled:
            logger.debug(
                f"Updated last_request_time: {datetime.fromtimestamp(self._last_request_time).isoformat()}"
            )

    @property
    def state(self) -> ModelState:
        """현재 모델 상태"""
        return self._state

    @property
    def is_active(self) -> bool:
        """모델이 활성 상태인지 확인"""
        return self._state == ModelState.ACTIVE

    @property
    def idle_seconds(self) -> Optional[float]:
        """마지막 요청 이후 경과 시간 (초)"""
        if self._last_request_time is None:
            return None
        return time.time() - self._last_request_time

    async def sleep(self):
        """모델을 슬립 모드로 전환 (Level 1)"""
        if not self.enabled:
            logger.debug("Sleep mode disabled, skipping")
            return

        if self._state == ModelState.SLEEPING:
            logger.debug("Model already sleeping")
            return

        if self._state == ModelState.WAKING:
            logger.debug("Model is waking up, cannot sleep now (race condition prevented)")
            return

        try:
            logger.info("Initiating sleep mode (Level 1)...")
            self._state = ModelState.SLEEPING

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(f"{self.vllm_dev_url}/sleep?level=1")
                response.raise_for_status()

            logger.info_event(
                "vllm_sleep",
                "vLLM model entered sleep mode (Level 1)",
                idle_seconds=self.idle_seconds
            )

        except Exception as e:
            logger.error(f"Failed to enter sleep mode: {e}")
            self._state = ModelState.ACTIVE  # 실패 시 ACTIVE로 복원
            raise

    async def wake(self):
        """모델을 깨우기"""
        if not self.enabled:
            logger.debug("Sleep mode disabled, skipping")
            return

        if self._state == ModelState.ACTIVE:
            logger.debug("Model already active")
            return

        try:
            logger.info("Waking up model from sleep...")
            self._state = ModelState.WAKING
            wake_start = time.time()

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(f"{self.vllm_dev_url}/wake_up")
                response.raise_for_status()

            wake_time = time.time() - wake_start
            self._state = ModelState.ACTIVE

            logger.info_event(
                "vllm_wake",
                "vLLM model woke up from sleep (Level 1)",
                duration_ms=wake_time * 1000
            )

        except Exception as e:
            logger.error(f"Failed to wake up model: {e}")
            self._state = ModelState.ACTIVE  # 실패해도 ACTIVE로 설정 (요청 시도)
            raise

    async def ensure_active(self):
        """모델이 활성 상태인지 확인하고, 슬립 중이면 깨우기"""
        if self._state == ModelState.SLEEPING:
            await self.wake()
        elif self._state == ModelState.WAKING:
            # 이미 깨어나는 중이면 대기
            logger.debug("Model is waking up, waiting...")
            while self._state == ModelState.WAKING:
                await asyncio.sleep(0.1)

    async def _check_idle_and_sleep(self):
        """Idle 상태 체크 및 슬립 (백그라운드 태스크)"""
        while True:
            try:
                await asyncio.sleep(self.check_interval)

                if not self.enabled:
                    continue

                # 아직 요청이 한 번도 없었으면 스킵
                if self._last_request_time is None:
                    logger.debug("No requests yet, skipping idle check")
                    continue

                # 이미 슬립 중이거나 깨어나는 중이면 스킵
                if self._state == ModelState.SLEEPING:
                    continue
                if self._state == ModelState.WAKING:
                    logger.debug("Model is waking up, skipping sleep check")
                    continue

                # Idle 시간 체크
                idle = self.idle_seconds
                if idle is not None and idle >= self.idle_timeout:
                    logger.info(
                        f"Model idle for {idle:.1f}s (threshold: {self.idle_timeout}s), "
                        f"entering sleep mode..."
                    )
                    await self.sleep()
                else:
                    logger.debug(
                        f"Model idle for {idle:.1f}s (threshold: {self.idle_timeout}s), "
                        f"staying active"
                    )

            except asyncio.CancelledError:
                logger.info("Sleep manager background task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in sleep manager background task: {e}")
                # 에러 발생 시에도 태스크는 계속 실행

    def start_background_task(self):
        """백그라운드 태스크 시작"""
        if not self.enabled:
            logger.info("Sleep mode disabled, background task not started")
            return

        if self._background_task is not None:
            logger.warning("Background task already running")
            return

        self._background_task = asyncio.create_task(self._check_idle_and_sleep())
        logger.info("Sleep manager background task started")

    def stop_background_task(self):
        """백그라운드 태스크 중지"""
        if self._background_task is not None:
            self._background_task.cancel()
            self._background_task = None
            logger.info("Sleep manager background task stopped")


# 전역 인스턴스 (싱글톤 패턴)
_sleep_manager: Optional[VLLMSleepManager] = None


def initialize_sleep_manager(
    vllm_dev_url: str,
    idle_timeout: int = 300,
    check_interval: int = 60,
    enabled: bool = True
) -> VLLMSleepManager:
    """슬립 매니저 초기화 (main.py에서 호출)"""
    global _sleep_manager
    _sleep_manager = VLLMSleepManager(
        vllm_dev_url=vllm_dev_url,
        idle_timeout=idle_timeout,
        check_interval=check_interval,
        enabled=enabled
    )
    return _sleep_manager


def get_sleep_manager() -> Optional[VLLMSleepManager]:
    """슬립 매니저 인스턴스 가져오기"""
    return _sleep_manager
