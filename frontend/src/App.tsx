import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap/dist/js/bootstrap.bundle.min.js';
import './modern-theme.css';
import React from 'react';
import { Route, BrowserRouter as Router, Routes } from 'react-router-dom';
import Navbar from './components/Navbar';
import History from './pages/History';
import Home from './pages/Home';

const App: React.FC = () => {
  return (
    <Router>
  <div className="min-vh-100" style={{ background: 'var(--primary-bg)' }}>
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/history" element={<History />} />
        </Routes>
        
        <footer className="mt-5 py-4 border-top" style={{ background: 'var(--secondary-bg)', color: 'var(--text-secondary)', borderRadius: 'var(--border-radius)' }}>
          <div className="container text-center text-read">
            <p className="mb-0 text-read" style={{ color: 'var(--accent2)' }}>
              ASL Classifier - Built with React, FastAPI, and TensorFlow
            </p>
          </div>
        </footer>
      </div>
    </Router>
  );
};

export default App;
