from __future__ import annotations

import asyncio

import v2_router as router_mod


class _Repo:
    def list_bitget_accounts(self):
        return [
            {
                "account_id": "acc1",
                "account_name": "main",
                "api_key_enc": "x",
                "api_secret_enc": "y",
                "passphrase_enc": "z",
                "enabled": True,
            }
        ]

    def list_proxy_profiles(self, include_disabled=True):  # noqa: ANN001
        _ = include_disabled
        return [
            {
                "profile_id": "p1",
                "name": "p",
                "proxy_type": "http",
                "host": "127.0.0.1",
                "port": 7890,
                "password_enc": "enc",
            }
        ]


def test_secrets_not_returned_from_listing_endpoints(monkeypatch):
    monkeypatch.setattr(router_mod, "repo", _Repo(), raising=False)
    accounts = asyncio.run(router_mod.list_live_accounts())
    assert accounts["items"]
    row = accounts["items"][0]
    assert "api_key_enc" not in row
    assert "api_secret_enc" not in row
    assert "passphrase_enc" not in row

    proxies = asyncio.run(router_mod.list_proxy_profiles(include_disabled=1))
    assert proxies["items"]
    prow = proxies["items"][0]
    assert "password_enc" not in prow
