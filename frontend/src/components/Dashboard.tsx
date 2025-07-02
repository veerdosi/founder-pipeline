import React from 'react'
import { useQuery } from '@tanstack/react-query'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import { TrendingUp, Users, Building2, Award } from 'lucide-react'

interface DashboardStats {
  totalCompanies: number
  totalFounders: number
  rankedFounders: number
  avgConfidenceScore: number
  levelDistribution: Record<string, number>
  recentActivity: Array<{
    id: string
    type: 'discovery' | 'ranking'
    timestamp: string
    count: number
  }>
}

const Dashboard: React.FC = () => {
  const { data: stats, isLoading } = useQuery<DashboardStats>({
    queryKey: ['dashboard-stats'],
    queryFn: async () => {
      const response = await fetch('/api/dashboard/stats')
      if (!response.ok) throw new Error('Failed to fetch stats')
      return response.json()
    }
  })

  const levelColors = {
    'L1': '#ef4444', 'L2': '#f97316', 'L3': '#f59e0b',
    'L4': '#eab308', 'L5': '#84cc16', 'L6': '#22c55e',
    'L7': '#10b981', 'L8': '#14b8a6', 'L9': '#06b6d4', 'L10': '#3b82f6'
  }

  if (isLoading) {
    return (
      <div className="p-8">
        <div className="animate-pulse space-y-6">
          <div className="h-8 bg-secondary-200 rounded w-1/4"></div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-24 bg-secondary-200 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-secondary-900">Dashboard</h1>
        <p className="text-secondary-600 mt-2">AI-powered founder discovery and ranking overview</p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="card p-6">
          <div className="flex items-center">
            <Building2 className="h-8 w-8 text-blue-600" />
            <div className="ml-4">
              <p className="text-sm font-medium text-secondary-600">Companies Discovered</p>
              <p className="text-2xl font-bold text-secondary-900">{stats?.totalCompanies || 0}</p>
            </div>
          </div>
        </div>

        <div className="card p-6">
          <div className="flex items-center">
            <Users className="h-8 w-8 text-green-600" />
            <div className="ml-4">
              <p className="text-sm font-medium text-secondary-600">Founders Analyzed</p>
              <p className="text-2xl font-bold text-secondary-900">{stats?.totalFounders || 0}</p>
            </div>
          </div>
        </div>

        <div className="card p-6">
          <div className="flex items-center">
            <Award className="h-8 w-8 text-purple-600" />
            <div className="ml-4">
              <p className="text-sm font-medium text-secondary-600">High Confidence Rankings</p>
              <p className="text-2xl font-bold text-secondary-900">{stats?.rankedFounders || 0}</p>
            </div>
          </div>
        </div>

        <div className="card p-6">
          <div className="flex items-center">
            <TrendingUp className="h-8 w-8 text-orange-600" />
            <div className="ml-4">
              <p className="text-sm font-medium text-secondary-600">Avg Confidence</p>
              <p className="text-2xl font-bold text-secondary-900">
                {stats?.avgConfidenceScore ? `${(stats.avgConfidenceScore * 100).toFixed(1)}%` : '0%'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Level Distribution */}
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-secondary-900 mb-4">Founder Level Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={Object.entries(stats?.levelDistribution || {}).map(([level, count]) => ({ level, count }))}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="level" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Recent Activity */}
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-secondary-900 mb-4">Recent Activity</h3>
          <div className="space-y-4">
            {stats?.recentActivity?.map((activity) => (
              <div key={activity.id} className="flex items-center justify-between p-3 bg-secondary-50 rounded-lg">
                <div>
                  <p className="font-medium text-secondary-900 capitalize">{activity.type}</p>
                  <p className="text-sm text-secondary-600">{new Date(activity.timestamp).toLocaleString()}</p>
                </div>
                <span className="bg-primary-100 text-primary-800 px-2 py-1 rounded-full text-sm">
                  {activity.count} items
                </span>
              </div>
            )) || (
              <p className="text-secondary-500 text-center py-8">No recent activity</p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default Dashboard
