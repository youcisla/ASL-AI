import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const Navbar: React.FC = () => {
  const location = useLocation();

  return (
  <nav className="navbar navbar-expand-lg" style={{ boxShadow: 'var(--shadow)', borderRadius: 'var(--border-radius)', background: 'var(--secondary-bg)' }}>
  <div className="container">
        <Link className="navbar-brand" to="/" style={{ color: 'var(--accent)', fontWeight: 700, fontSize: '1.7rem', letterSpacing: '1px' }}>
          <strong>ASL Classifier</strong>
        </Link>
        
        <button
          className="navbar-toggler"
          type="button"
          data-bs-toggle="collapse"
          data-bs-target="#navbarNav"
          aria-controls="navbarNav"
          aria-expanded="false"
          aria-label="Toggle navigation"
        >
          <span className="navbar-toggler-icon"></span>
        </button>
        
        <div className="collapse navbar-collapse" id="navbarNav">
          <ul className="navbar-nav ms-auto">
            <li className="nav-item">
              <Link
                className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}
                to="/"
              >
                Home
              </Link>
            </li>
            <li className="nav-item">
              <Link
                className={`nav-link ${location.pathname === '/history' ? 'active' : ''}`}
                to="/history"
              >
                History
              </Link>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
