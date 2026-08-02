"""
Slunder Studio — OS Credential Store
Secrets live in the operating system credential service, never in config JSON,
logs, diagnostics, or timestamped backups.

Backend order:
  1. the `keyring` package when it resolves a real backend
  2. Windows Credential Manager via advapi32
  3. macOS Keychain via the `security` tool
  4. Linux Secret Service via `secret-tool`

When none is usable the store reports an explicit unavailable state instead of
silently falling back to plaintext.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import Optional

SERVICE_NAME = "SlunderStudio"
HF_TOKEN_ACCOUNT = "huggingface-token"

# Windows: run child processes without flashing a console window.
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


class CredentialError(RuntimeError):
    """Raised when a secret cannot be stored or removed."""


@dataclass(frozen=True)
class CredentialBackendStatus:
    """What the credential service can do right now, and why."""
    available: bool
    name: str
    detail: str

    def as_dict(self) -> dict:
        return {"available": self.available, "name": self.name, "detail": self.detail}


class _Backend:
    name = "unavailable"

    def available(self) -> bool:
        return False

    def detail(self) -> str:
        return "No OS credential service is available."

    def get(self, service: str, account: str) -> Optional[str]:
        raise NotImplementedError

    def set(self, service: str, account: str, secret: str) -> None:
        raise NotImplementedError

    def delete(self, service: str, account: str) -> bool:
        raise NotImplementedError


class _KeyringBackend(_Backend):
    """Uses the `keyring` package when it resolves to a real backend."""

    name = "keyring"

    def __init__(self):
        self._keyring = None
        self._detail = "The keyring package is not installed."
        try:
            import keyring
            from keyring.backends import fail as keyring_fail
        except Exception as exc:  # pragma: no cover - depends on environment
            self._detail = f"The keyring package is unavailable: {exc}"
            return
        try:
            backend = keyring.get_keyring()
        except Exception as exc:  # pragma: no cover - depends on environment
            self._detail = f"keyring could not resolve a backend: {exc}"
            return
        if isinstance(backend, keyring_fail.Keyring):
            self._detail = "keyring resolved no usable backend on this system."
            return
        self._keyring = keyring
        self.name = f"keyring ({type(backend).__name__})"
        self._detail = f"Using {self.name}."

    def available(self) -> bool:
        return self._keyring is not None

    def detail(self) -> str:
        return self._detail

    def get(self, service: str, account: str) -> Optional[str]:
        return self._keyring.get_password(service, account) or None

    def set(self, service: str, account: str, secret: str) -> None:
        self._keyring.set_password(service, account, secret)

    def delete(self, service: str, account: str) -> bool:
        try:
            self._keyring.delete_password(service, account)
            return True
        except Exception:
            return False


class _WindowsBackend(_Backend):
    """Windows Credential Manager through advapi32, no extra dependency."""

    name = "Windows Credential Manager"

    CRED_TYPE_GENERIC = 1
    CRED_PERSIST_LOCAL_MACHINE = 2
    ERROR_NOT_FOUND = 1168

    def __init__(self):
        self._ok = False
        self._detail = "Windows Credential Manager is only available on Windows."
        if os.name != "nt":
            return
        try:
            import ctypes
            from ctypes import wintypes
        except Exception as exc:  # pragma: no cover - ctypes is stdlib
            self._detail = f"ctypes is unavailable: {exc}"
            return

        class FILETIME(ctypes.Structure):
            _fields_ = [
                ("dwLowDateTime", wintypes.DWORD),
                ("dwHighDateTime", wintypes.DWORD),
            ]

        class CREDENTIAL_ATTRIBUTEW(ctypes.Structure):
            _fields_ = [
                ("Keyword", wintypes.LPWSTR),
                ("Flags", wintypes.DWORD),
                ("ValueSize", wintypes.DWORD),
                ("Value", ctypes.POINTER(ctypes.c_byte)),
            ]

        class CREDENTIALW(ctypes.Structure):
            _fields_ = [
                ("Flags", wintypes.DWORD),
                ("Type", wintypes.DWORD),
                ("TargetName", wintypes.LPWSTR),
                ("Comment", wintypes.LPWSTR),
                ("LastWritten", FILETIME),
                ("CredentialBlobSize", wintypes.DWORD),
                ("CredentialBlob", ctypes.POINTER(ctypes.c_byte)),
                ("Persist", wintypes.DWORD),
                ("AttributeCount", wintypes.DWORD),
                ("Attributes", ctypes.POINTER(CREDENTIAL_ATTRIBUTEW)),
                ("TargetAlias", wintypes.LPWSTR),
                ("UserName", wintypes.LPWSTR),
            ]

        self._ctypes = ctypes
        self._CREDENTIALW = CREDENTIALW
        try:
            self._advapi = ctypes.WinDLL("advapi32", use_last_error=True)
        except Exception as exc:  # pragma: no cover - depends on environment
            self._detail = f"advapi32 could not be loaded: {exc}"
            return
        self._advapi.CredReadW.argtypes = [
            wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
            ctypes.POINTER(ctypes.POINTER(CREDENTIALW)),
        ]
        self._advapi.CredReadW.restype = wintypes.BOOL
        self._advapi.CredWriteW.argtypes = [
            ctypes.POINTER(CREDENTIALW), wintypes.DWORD,
        ]
        self._advapi.CredWriteW.restype = wintypes.BOOL
        self._advapi.CredDeleteW.argtypes = [
            wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
        ]
        self._advapi.CredDeleteW.restype = wintypes.BOOL
        self._advapi.CredFree.argtypes = [ctypes.c_void_p]
        self._advapi.CredFree.restype = None
        self._ok = True
        self._detail = "Using Windows Credential Manager."

    @staticmethod
    def _target(service: str, account: str) -> str:
        return f"{service}:{account}"

    def available(self) -> bool:
        return self._ok

    def detail(self) -> str:
        return self._detail

    def get(self, service: str, account: str) -> Optional[str]:
        ctypes = self._ctypes
        pointer = ctypes.POINTER(self._CREDENTIALW)()
        ok = self._advapi.CredReadW(
            self._target(service, account), self.CRED_TYPE_GENERIC, 0,
            ctypes.byref(pointer),
        )
        if not ok:
            code = ctypes.get_last_error()
            if code == self.ERROR_NOT_FOUND:
                return None
            raise CredentialError(f"CredReadW failed with error {code}")
        try:
            cred = pointer.contents
            size = int(cred.CredentialBlobSize)
            if size <= 0:
                return None
            blob = ctypes.string_at(cred.CredentialBlob, size)
            return blob.decode("utf-16-le") or None
        finally:
            self._advapi.CredFree(pointer)

    def set(self, service: str, account: str, secret: str) -> None:
        ctypes = self._ctypes
        blob = secret.encode("utf-16-le")
        buffer = ctypes.create_string_buffer(blob, len(blob))
        cred = self._CREDENTIALW()
        cred.Flags = 0
        cred.Type = self.CRED_TYPE_GENERIC
        cred.TargetName = self._target(service, account)
        cred.Comment = f"{service} secret"
        cred.CredentialBlobSize = len(blob)
        cred.CredentialBlob = ctypes.cast(
            buffer, ctypes.POINTER(ctypes.c_byte)
        )
        cred.Persist = self.CRED_PERSIST_LOCAL_MACHINE
        cred.AttributeCount = 0
        cred.Attributes = None
        cred.TargetAlias = None
        cred.UserName = account
        if not self._advapi.CredWriteW(ctypes.byref(cred), 0):
            raise CredentialError(
                f"CredWriteW failed with error {ctypes.get_last_error()}"
            )

    def delete(self, service: str, account: str) -> bool:
        ok = self._advapi.CredDeleteW(
            self._target(service, account), self.CRED_TYPE_GENERIC, 0
        )
        return bool(ok)


class _CommandBackend(_Backend):
    """Shared plumbing for CLI-driven keychains."""

    tool = ""

    def __init__(self):
        self._path = shutil.which(self.tool) if self.tool else None
        self._detail = (
            f"Using {self.name}." if self._path
            else f"`{self.tool}` was not found on PATH."
        )

    def available(self) -> bool:
        return bool(self._path)

    def detail(self) -> str:
        return self._detail

    def _run(self, args: list[str], stdin: Optional[str] = None):
        return subprocess.run(
            [self._path, *args],
            input=stdin,
            capture_output=True,
            text=True,
            creationflags=_NO_WINDOW,
        )


class _MacKeychainBackend(_CommandBackend):
    name = "macOS Keychain"
    tool = "security"

    def get(self, service: str, account: str) -> Optional[str]:
        proc = self._run(
            ["find-generic-password", "-s", service, "-a", account, "-w"]
        )
        if proc.returncode != 0:
            return None
        return proc.stdout.strip() or None

    def set(self, service: str, account: str, secret: str) -> None:
        proc = self._run(
            ["add-generic-password", "-U", "-s", service, "-a", account,
             "-w", secret]
        )
        if proc.returncode != 0:
            raise CredentialError(
                f"security add-generic-password failed: {proc.stderr.strip()}"
            )

    def delete(self, service: str, account: str) -> bool:
        proc = self._run(
            ["delete-generic-password", "-s", service, "-a", account]
        )
        return proc.returncode == 0


class _SecretToolBackend(_CommandBackend):
    name = "Linux Secret Service"
    tool = "secret-tool"

    def get(self, service: str, account: str) -> Optional[str]:
        proc = self._run(["lookup", "service", service, "account", account])
        if proc.returncode != 0:
            return None
        return proc.stdout.strip() or None

    def set(self, service: str, account: str, secret: str) -> None:
        proc = self._run(
            ["store", "--label", f"{service} {account}",
             "service", service, "account", account],
            stdin=secret,
        )
        if proc.returncode != 0:
            raise CredentialError(
                f"secret-tool store failed: {proc.stderr.strip()}"
            )

    def delete(self, service: str, account: str) -> bool:
        proc = self._run(["clear", "service", service, "account", account])
        return proc.returncode == 0


def _build_backend() -> _Backend:
    candidates: list[_Backend] = [_KeyringBackend()]
    system = platform.system()
    if system == "Windows":
        candidates.append(_WindowsBackend())
    elif system == "Darwin":
        candidates.append(_MacKeychainBackend())
    else:
        candidates.append(_SecretToolBackend())
    for backend in candidates:
        if backend.available():
            return backend
    unavailable = _Backend()
    reasons = "; ".join(b.detail() for b in candidates)
    unavailable_detail = (
        "No OS credential service is available. Secrets will not be stored. "
        f"({reasons})"
    )
    unavailable.detail = lambda: unavailable_detail  # type: ignore[method-assign]
    return unavailable


class CredentialStore:
    """Reads and writes secrets in the OS credential service."""

    _instance: Optional["CredentialStore"] = None
    _singleton_lock = threading.Lock()

    def __new__(cls, service: str = SERVICE_NAME):
        if service != SERVICE_NAME:
            return super().__new__(cls)
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, service: str = SERVICE_NAME):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._service = service
        self._lock = threading.RLock()
        self._backend = _build_backend()

    # ── Backend state ──────────────────────────────────────────────────────────

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def status(self) -> CredentialBackendStatus:
        return CredentialBackendStatus(
            available=self._backend.available(),
            name=self._backend.name,
            detail=self._backend.detail(),
        )

    def is_available(self) -> bool:
        return self._backend.available()

    # ── Secrets ────────────────────────────────────────────────────────────────

    def get_secret(self, account: str) -> Optional[str]:
        """Return the stored secret, or None when absent or unreadable."""
        with self._lock:
            if not self._backend.available():
                return None
            try:
                return self._backend.get(self._service, account) or None
            except Exception:
                return None

    def set_secret(self, account: str, secret: str) -> None:
        """Store a secret. Raises CredentialError when no backend can hold it."""
        secret = (secret or "").strip()
        if not secret:
            self.delete_secret(account)
            return
        with self._lock:
            if not self._backend.available():
                raise CredentialError(self._backend.detail())
            self._backend.set(self._service, account, secret)

    def delete_secret(self, account: str) -> bool:
        """Remove a secret. Returns True when something was removed."""
        with self._lock:
            if not self._backend.available():
                return False
            try:
                return bool(self._backend.delete(self._service, account))
            except Exception:
                return False


class MemoryCredentialStore(CredentialStore):
    """In-process credential store. Used by tests so they never touch the real
    OS credential service."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, service: str = SERVICE_NAME, available: bool = True):
        self._initialized = True
        self._service = service
        self._lock = threading.RLock()
        self._available = available
        self._secrets: dict[str, str] = {}
        self._backend = _Backend()

    @property
    def backend_name(self) -> str:
        return "in-memory credential store" if self._available else "unavailable"

    def status(self) -> CredentialBackendStatus:
        return CredentialBackendStatus(
            available=self._available,
            name=self.backend_name,
            detail=(
                "Using the in-memory credential store."
                if self._available
                else "No OS credential service is available."
            ),
        )

    def is_available(self) -> bool:
        return self._available

    def get_secret(self, account: str) -> Optional[str]:
        with self._lock:
            if not self._available:
                return None
            return self._secrets.get(account) or None

    def set_secret(self, account: str, secret: str) -> None:
        secret = (secret or "").strip()
        with self._lock:
            if not self._available:
                raise CredentialError("No OS credential service is available.")
            if secret:
                self._secrets[account] = secret
            else:
                self._secrets.pop(account, None)

    def delete_secret(self, account: str) -> bool:
        with self._lock:
            return self._secrets.pop(account, None) is not None


def get_credential_store() -> CredentialStore:
    return CredentialStore()
