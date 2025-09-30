import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import Home from './pages/Home';
import Architecture from './pages/Architecture';
import Results from './pages/Results';
import Implementation from './pages/Implementation';
import InteractiveDemo from './pages/InteractiveDemo';

function App() {
  return (
    <Router>
      <div className="App">
        <Header />
        <div className="main-content">
          <Sidebar />
          <main className="content">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/architecture" element={<Architecture />} />
              <Route path="/results" element={<Results />} />
              <Route path="/implementation" element={<Implementation />} />
              <Route path="/demo" element={<InteractiveDemo />} />
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  );
}

export default App;