# esp32_client.py
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import requests


class ESP32Client:
    """
    ESP32-C3 Web API クライアント（Session再利用 + 簡易リトライ付き）
    """

    def __init__(
        self,
        base_url: str = "http://192.168.4.1",
        timeout: float = 1.0,
        retries: int = 3,
        backoff_sec: float = 0.2,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.backoff_sec = backoff_sec

        self._session = requests.Session()

    def close(self) -> None:
        self._session.close()

    def _get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_err: Optional[Exception] = None

        for i in range(self.retries):
            try:
                r = self._session.get(url, params=params, timeout=self.timeout)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                # 軽いバックオフ
                time.sleep(self.backoff_sec * (2 ** i))

        raise RuntimeError(f"GET failed: {url} params={params} err={last_err}") from last_err

    def get_status(self) -> Dict[str, Any]:
        return self._get_json("/api/status")

    def get_sensors(self) -> Dict[str, Any]:
        """
        /api/sensors を取得
        例:
        {
          "timestamp": 12345678,
          "heart_rate": 72,
          "spo2": 98,
          "temperature": 36.5,
          "Pitch": 10.25,
          "Yaw": -5.30,
          "Roll": 2.15
        }
        """
        return self._get_json("/api/sensors")

    def set_pwm(self, value: float) -> Dict[str, Any]:
        """
        /api/pwm?value=XX
        """
        return self._get_json("/api/pwm", params={"value": float(value)})
