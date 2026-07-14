"""Frontend-only authentication shell for the Streamlit dashboard."""

from datetime import datetime
import re
import time

import streamlit as st


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _display_name(email: str) -> str:
    local = email.split("@", 1)[0].replace(".", " ").replace("_", " ")
    return " ".join(part.capitalize() for part in local.split()) or "Research User"


def _initials(name: str) -> str:
    value = "".join(part[0].upper() for part in name.split()[:2] if part)
    return value or "EI"


def _login(email: str, remember: bool, guest: bool = False) -> None:
    name = "Guest Analyst" if guest else _display_name(email)
    st.session_state.is_authenticated = True
    st.session_state.remember_me = remember
    st.session_state.last_login = datetime.now().strftime("%d %b %Y, %I:%M %p")
    st.session_state.auth_user = {
        "name": name,
        "email": email,
        "avatar": _initials(name),
        "role": "Guest Session" if guest else "Research Analyst",
    }
    if not st.session_state.get("welcome_dismissed", False):
        st.session_state.show_welcome = True


def logout_user() -> None:
    st.session_state.is_authenticated = False
    st.session_state.auth_user = None
    st.session_state.show_welcome = False
    st.session_state.active_page = "Dashboard"


def render_login_page() -> None:
    st.markdown(
        """
        <div class="login-hero">
            <div class="login-grid"></div>
            <div class="login-brand">
                <div class="login-mark">EI</div>
                <div class="login-kicker">Secure Scientific Intelligence</div>
                <h1>Earth Intelligence Platform</h1>
                <p>Secure Multi-Agent Scientific Intelligence System</p>
                <div class="login-chips">
                    <span>Planner Agent</span>
                    <span>Dataset Discovery</span>
                    <span>Security Validation</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns([1.15, 1, 1.15])
    with center:
        st.markdown(
            """
            <div class="login-card-title">
                <div class="login-card-logo">EI</div>
                <div>
                    <h2>Welcome back</h2>
                    <p>Sign in to continue to your dashboard.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("Email", placeholder="researcher@example.com")
            show_password = st.checkbox("Show password")
            password = st.text_input(
                "Password",
                placeholder="Enter your password",
                type="default" if show_password else "password",
            )
            remember = st.checkbox("Remember Me", value=st.session_state.get("remember_me", False))
            submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

        left, right = st.columns(2)
        with left:
            forgot = st.button("Forgot Password", use_container_width=True)
        with right:
            guest = st.button("Continue as Guest", use_container_width=True)

        if forgot:
            st.info("Password recovery is ready for backend integration.")

        if submitted:
            errors = []
            if not EMAIL_RE.match(email.strip()):
                errors.append("Enter a valid email address.")
            if len(password or "") < 6:
                errors.append("Password must be at least 6 characters.")
            if errors:
                for error in errors:
                    st.error(error)
            else:
                with st.spinner("Signing in..."):
                    time.sleep(0.6)
                _login(email.strip(), remember)
                st.success("Login successful. Opening dashboard...")
                time.sleep(0.25)
                st.rerun()

        if guest:
            with st.spinner("Preparing guest workspace..."):
                time.sleep(0.4)
            _login("guest@earth-intelligence.local", False, guest=True)
            st.success("Guest session started.")
            time.sleep(0.25)
            st.rerun()


def render_profile_header() -> None:
    user = st.session_state.get("auth_user") or {}
    name = user.get("name", "Research User")
    email = user.get("email", "user@example.com")
    avatar = user.get("avatar", "EI")
    last_login = st.session_state.get("last_login") or "Current session"

    left, right = st.columns([4.8, 2])
    with left:
        st.markdown(
            """
            <div class="platform-header">
                <div class="platform-mark">EI</div>
                <div>
                    <h1>Earth Intelligence Platform</h1>
                    <p>Enterprise multi-agent scientific intelligence workspace</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"""
            <div class="profile-pill">
                <div class="profile-avatar">{avatar}</div>
                <div class="profile-copy">
                    <strong>{name}</strong>
                    <span>{email}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Profile Menu", expanded=False):
            st.caption(f"Role: {user.get('role', 'Research Analyst')}")
            st.caption(f"Last Login: {last_login}")
            settings_col, logout_col = st.columns(2)
            with settings_col:
                if st.button("Settings", use_container_width=True):
                    st.session_state.active_page = "Settings"
                    st.rerun()
            with logout_col:
                if st.button("Logout", use_container_width=True):
                    logout_user()
                    st.rerun()


def render_welcome_modal() -> None:
    if not st.session_state.get("show_welcome", False):
        return
    st.markdown(
        """
        <div class="welcome-panel">
            <h3>Welcome to Earth Intelligence</h3>
            <ol>
                <li>Enter your scientific query.</li>
                <li>The Planner Agent creates a research strategy.</li>
                <li>The Clarification Agent refines the request.</li>
                <li>Dataset Discovery locates relevant scientific data.</li>
                <li>Security Validation verifies integrity.</li>
                <li>The Final Intelligence Report is generated.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )
    dont_show = st.checkbox("Don't show again", key="dont_show_welcome")
    if st.button("Start Analysis", type="primary"):
        st.session_state.show_welcome = False
        if dont_show:
            st.session_state.welcome_dismissed = True
        st.rerun()
