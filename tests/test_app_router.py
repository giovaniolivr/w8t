from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from streamlit.testing.v1 import AppTest

import w8t.data.db as db_module
from w8t.data.models import Base

APP = Path(__file__).resolve().parents[1] / "src" / "w8t" / "app"


@pytest.fixture
def app(tmp_path, monkeypatch):
    test_engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(test_engine)
    monkeypatch.setattr(db_module, "engine", test_engine)
    monkeypatch.setattr(
        db_module, "SessionLocal", sessionmaker(bind=test_engine, expire_on_commit=False)
    )
    return AppTest.from_file(str(APP / "app.py"), default_timeout=30)


def test_router_opens_the_dashboard_by_default(app):
    at = app.run()

    assert not at.exception
    assert any("Dashboard" in m.value for m in at.markdown)


def test_every_registered_page_exists_and_legacy_pages_dir_is_gone():
    import ast

    tree = ast.parse((APP / "app.py").read_text(encoding="utf-8"))
    files = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and node.value.endswith(".py")
    ]
    assert len(files) == 6
    for name in files:
        assert list(APP.rglob(name)), name
    # A pages/ dir would be auto-registered by Streamlit and clash with st.navigation.
    assert not (APP / "pages").exists()
