import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap/dist/js/bootstrap.bundle.min.js';
import React from 'react';
import { Route, BrowserRouter as Router, Routes } from 'react-router-dom';
import Navbar from './components/Navbar';
import History from './pages/History';
import Home from './pages/Home';

const App: React.FC = () => {
  return (
    <Router>
      <div className="min-vh-100 bg-light">
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/history" element={<History />} />
        </Routes>
        
        <footer className="mt-5 py-4 bg-white border-top">
          <div className="container text-center">
            <p className="text-muted mb-0">
              ASL Classifier - Built with React, FastAPI, and TensorFlow
            </p>
          </div>
        </footer>
      </div>
    </Router>
  );
};

export default App;
