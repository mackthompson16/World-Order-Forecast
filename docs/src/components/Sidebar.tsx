import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Home, Cpu, BarChart3, Code, PlayCircle } from 'lucide-react';
import './Sidebar.css';

const Sidebar: React.FC = () => {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Overview', icon: Home },
    { path: '/architecture', label: 'Architecture', icon: Cpu },
    { path: '/results', label: 'Results', icon: BarChart3 },
    { path: '/implementation', label: 'Implementation', icon: Code },
    { path: '/demo', label: 'Interactive Demo', icon: PlayCircle },
  ];

  return (
    <aside className="sidebar">
      <nav className="sidebar-nav">
        {navItems.map(({ path, label, icon: Icon }) => (
          <Link
            key={path}
            to={path}
            className={`nav-item ${location.pathname === path ? 'active' : ''}`}
          >
            <Icon size={20} />
            <span>{label}</span>
          </Link>
        ))}
      </nav>
      
      <div className="sidebar-footer">
        <div className="version-info">
          <span className="version-label">Version</span>
          <span className="version-number">1.0.0</span>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
