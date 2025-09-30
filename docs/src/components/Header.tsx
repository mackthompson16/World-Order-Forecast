import React from 'react';
import { Link } from 'react-router-dom';
import { Globe, Github, ExternalLink } from 'lucide-react';
import './Header.css';

const Header: React.FC = () => {
  return (
    <header className="header">
      <div className="header-content">
        <Link to="/" className="logo">
          <Globe className="logo-icon" />
          <span className="logo-text">Country Standing Forecast</span>
        </Link>
        
        <nav className="header-nav">
          <a 
            href="https://github.com/your-username/country-standing-forecast" 
            target="_blank" 
            rel="noopener noreferrer"
            className="nav-link"
          >
            <Github size={20} />
            <span>GitHub</span>
          </a>
          
          <a 
            href="https://your-username.github.io/country-standing-forecast" 
            target="_blank" 
            rel="noopener noreferrer"
            className="nav-link"
          >
            <ExternalLink size={20} />
            <span>Live Demo</span>
          </a>
        </nav>
      </div>
    </header>
  );
};

export default Header;
