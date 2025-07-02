import React from 'react'
import { NavLink } from 'react-router-dom'
import { BarChart3, Search, Trophy, Home } from 'lucide-react'

const Sidebar: React.FC = () => {
  const navItems = [
    { path: '/', icon: Home, label: 'Dashboard' },
    { path: '/discovery', icon: Search, label: 'Company Discovery' },
    { path: '/ranking', icon: Trophy, label: 'Founder Ranking' },
  ]

  return (
    <div className="w-64 bg-white shadow-sm border-r border-secondary-200">
      <div className="p-6">
        <div className="flex items-center">
          <BarChart3 className="h-8 w-8 text-primary-600" />
          <h1 className="ml-3 text-xl font-bold text-secondary-900">
            Initiation Pipeline
          </h1>
        </div>
      </div>

      <nav className="mt-8">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              `flex items-center px-6 py-3 text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-primary-50 text-primary-700 border-r-2 border-primary-600'
                  : 'text-secondary-600 hover:text-secondary-900 hover:bg-secondary-50'
              }`
            }
          >
            <item.icon className="h-5 w-5 mr-3" />
            {item.label}
          </NavLink>
        ))}
      </nav>

      <div className="absolute bottom-0 w-64 p-6 border-t border-secondary-200">
        <div className="text-xs text-secondary-500">
          AI-Powered Founder Discovery
        </div>
      </div>
    </div>
  )
}

export default Sidebar
