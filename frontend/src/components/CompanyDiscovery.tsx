import React, { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Play, Download, Settings, RefreshCw } from 'lucide-react'

interface CompanyDiscoveryParams {
  limit: number
  categories: string[]
  regions: string[]
  sources: string[]
}

interface DiscoveredCompany {
  id: string
  name: string
  description: string
  website?: string
  foundedYear?: number
  fundingTotal?: number
  fundingStage?: string
  founders: string[]
  location: string
  aiCategory: string
  source: string
}

const CompanyDiscovery: React.FC = () => {
  const [params, setParams] = useState<CompanyDiscoveryParams>({
    limit: 50,
    categories: [],
    regions: [],
    sources: ['techcrunch', 'crunchbase', 'ycombinator']
  })

  const queryClient = useQueryClient()

  const { data: companies, isLoading } = useQuery<DiscoveredCompany[]>({
    queryKey: ['companies'],
    queryFn: async () => {
      const response = await fetch('/api/companies')
      if (!response.ok) throw new Error('Failed to fetch companies')
      return response.json()
    }
  })

  const discoveryMutation = useMutation({
    mutationFn: async (params: CompanyDiscoveryParams) => {
      const response = await fetch('/api/companies/discover', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      })
      if (!response.ok) throw new Error('Discovery failed')
      return response.json()
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['companies'] })
    }
  })

  const exportMutation = useMutation({
    mutationFn: async () => {
      const response = await fetch('/api/companies/export')
      if (!response.ok) throw new Error('Export failed')
      return response.blob()
    },
    onSuccess: (blob) => {
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `companies_${new Date().toISOString().split('T')[0]}.csv`
      a.click()
      window.URL.revokeObjectURL(url)
    }
  })

  return (
    <div className="p-8 space-y-8">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-secondary-900">Company Discovery</h1>
          <p className="text-secondary-600 mt-2">Discover AI companies from multiple sources</p>
        </div>
        <div className="flex space-x-3">
          <button
            onClick={() => exportMutation.mutate()}
            disabled={exportMutation.isPending || !companies?.length}
            className="btn-secondary flex items-center"
          >
            <Download className="h-4 w-4 mr-2" />
            Export CSV
          </button>
          <button
            onClick={() => discoveryMutation.mutate(params)}
            disabled={discoveryMutation.isPending}
            className="btn-primary flex items-center"
          >
            {discoveryMutation.isPending ? (
              <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Play className="h-4 w-4 mr-2" />
            )}
            Start Discovery
          </button>
        </div>
      </div>

      {/* Discovery Parameters */}
      <div className="card p-6">
        <h3 className="text-lg font-semibold text-secondary-900 mb-4">Discovery Parameters</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <label className="block text-sm font-medium text-secondary-700 mb-2">
              Company Limit
            </label>
            <input
              type="number"
              value={params.limit}
              onChange={(e) => setParams({ ...params, limit: parseInt(e.target.value) })}
              className="input-field"
              min="1"
              max="500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-secondary-700 mb-2">
              AI Categories
            </label>
            <select
              multiple
              value={params.categories}
              onChange={(e) => setParams({ 
                ...params, 
                categories: Array.from(e.target.selectedOptions, option => option.value)
              })}
              className="input-field h-20"
            >
              <option value="machine-learning">Machine Learning</option>
              <option value="computer-vision">Computer Vision</option>
              <option value="nlp">Natural Language Processing</option>
              <option value="robotics">Robotics</option>
              <option value="autonomous-vehicles">Autonomous Vehicles</option>
              <option value="healthcare-ai">Healthcare AI</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-secondary-700 mb-2">
              Data Sources
            </label>
            <div className="space-y-2">
              {['techcrunch', 'crunchbase', 'ycombinator', 'techstars', 'angellist'].map((source) => (
                <label key={source} className="flex items-center">
                  <input
                    type="checkbox"
                    checked={params.sources.includes(source)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setParams({ ...params, sources: [...params.sources, source] })
                      } else {
                        setParams({ ...params, sources: params.sources.filter(s => s !== source) })
                      }
                    }}
                    className="mr-2"
                  />
                  {source.charAt(0).toUpperCase() + source.slice(1)}
                </label>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Results */}
      <div className="card">
        <div className="p-6 border-b border-secondary-200">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold text-secondary-900">
              Discovered Companies ({companies?.length || 0})
            </h3>
            {discoveryMutation.isPending && (
              <div className="text-sm text-primary-600">Discovery in progress...</div>
            )}
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-secondary-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 uppercase">Name</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 uppercase">Category</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 uppercase">Funding</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 uppercase">Location</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 uppercase">Founders</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 uppercase">Source</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-secondary-200">
              {companies?.map((company) => (
                <tr key={company.id} className="hover:bg-secondary-50">
                  <td className="px-6 py-4">
                    <div>
                      <div className="font-medium text-secondary-900">{company.name}</div>
                      <div className="text-sm text-secondary-500 truncate max-w-xs">
                        {company.description}
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 text-sm text-secondary-900">{company.aiCategory}</td>
                  <td className="px-6 py-4 text-sm text-secondary-900">
                    {company.fundingTotal ? `$${(company.fundingTotal / 1000000).toFixed(1)}M` : 'N/A'}
                  </td>
                  <td className="px-6 py-4 text-sm text-secondary-900">{company.location}</td>
                  <td className="px-6 py-4 text-sm text-secondary-900">
                    {company.founders.slice(0, 2).join(', ')}
                    {company.founders.length > 2 && ` +${company.founders.length - 2}`}
                  </td>
                  <td className="px-6 py-4 text-sm text-secondary-500">{company.source}</td>
                </tr>
              )) || (
                <tr>
                  <td colSpan={6} className="px-6 py-8 text-center text-secondary-500">
                    {isLoading ? 'Loading companies...' : 'No companies discovered yet'}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

export default CompanyDiscovery
