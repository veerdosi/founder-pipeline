import React, { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Trophy, Play, Upload, Download, Filter } from 'lucide-react'

interface RankingParams {
  minConfidence: number
  batchSize: number
  sourceFile?: File
}

interface FounderRanking {
  id: string
  name: string
  company: string
  level: string
  confidenceScore: number
  reasoning: string
  evidence: string[]
  verificationSources: string[]
  timestamp: string
}

const FounderRanking: React.FC = () => {
  const [params, setParams] = useState<RankingParams>({
    minConfidence: 0.75,
    batchSize: 5
  })
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [filterLevel, setFilterLevel] = useState<string>('all')

  const queryClient = useQueryClient()

  const { data: rankings, isLoading } = useQuery<FounderRanking[]>({
    queryKey: ['founder-rankings'],
    queryFn: async () => {
      const response = await fetch('/api/founders/rankings')
      if (!response.ok) throw new Error('Failed to fetch rankings')
      return response.json()
    }
  })

  const rankingMutation = useMutation({
    mutationFn: async (params: RankingParams & { file?: File }) => {
      const formData = new FormData()
      formData.append('minConfidence', params.minConfidence.toString())
      formData.append('batchSize', params.batchSize.toString())
      if (params.file) formData.append('file', params.file)

      const response = await fetch('/api/founders/rank', {
        method: 'POST',
        body: formData
      })
      if (!response.ok) throw new Error('Ranking failed')
      return response.json()
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['founder-rankings'] })
    }
  })

  const levelColors: Record<string, string> = {
    'L1': 'bg-red-100 text-red-800',
    'L2': 'bg-orange-100 text-orange-800', 
    'L3': 'bg-yellow-100 text-yellow-800',
    'L4': 'bg-blue-100 text-blue-800',
    'L5': 'bg-green-100 text-green-800',
    'L6': 'bg-purple-100 text-purple-800',
    'L7': 'bg-pink-100 text-pink-800',
    'L8': 'bg-indigo-100 text-indigo-800',
    'L9': 'bg-cyan-100 text-cyan-800',
    'L10': 'bg-emerald-100 text-emerald-800',
    'INSUFFICIENT_DATA': 'bg-gray-100 text-gray-800'
  }

  const filteredRankings = rankings?.filter(ranking => 
    filterLevel === 'all' || ranking.level === filterLevel
  ) || []

  return (
    <div className="p-8 space-y-8">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-secondary-900">Founder Ranking</h1>
          <p className="text-secondary-600 mt-2">L1-L10 Carnegie Mellon framework analysis</p>
        </div>
        <div className="flex space-x-3">
          <button
            onClick={() => document.getElementById('file-upload')?.click()}
            className="btn-secondary flex items-center"
          >
            <Upload className="h-4 w-4 mr-2" />
            Upload CSV
          </button>
          <button
            onClick={() => rankingMutation.mutate(params)}
            disabled={rankingMutation.isPending}
            className="btn-primary flex items-center"
          >
            {rankingMutation.isPending ? (
              <div className="h-4 w-4 mr-2 animate-spin rounded-full border-2 border-white border-t-transparent" />
            ) : (
              <Play className="h-4 w-4 mr-2" />
            )}
            Start Ranking
          </button>
        </div>
      </div>

      <input
        id="file-upload"
        type="file"
        accept=".csv"
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) setUploadedFile(file)
        }}
        className="hidden"
      />

      {/* Configuration */}
      <div className="card p-6">
        <h3 className="text-lg font-semibold text-secondary-900 mb-4">Ranking Configuration</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <label className="block text-sm font-medium text-secondary-700 mb-2">
              Minimum Confidence ({(params.minConfidence * 100).toFixed(0)}%)
            </label>
            <input
              type="range"
              min="0.5"
              max="1.0"
              step="0.05"
              value={params.minConfidence}
              onChange={(e) => setParams({ ...params, minConfidence: parseFloat(e.target.value) })}
              className="w-full"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-secondary-700 mb-2">
              Batch Size
            </label>
            <select
              value={params.batchSize}
              onChange={(e) => setParams({ ...params, batchSize: parseInt(e.target.value) })}
              className="input-field"
            >
              <option value={3}>3 (Conservative)</option>
              <option value={5}>5 (Balanced)</option>
              <option value={10}>10 (Fast)</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-secondary-700 mb-2">
              Source File
            </label>
            <div className="text-sm text-secondary-600">
              {uploadedFile ? uploadedFile.name : 'No file selected'}
            </div>
          </div>
        </div>
      </div>

      {/* Results */}
      <div className="card">
        <div className="p-6 border-b border-secondary-200">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold text-secondary-900">
              Rankings ({filteredRankings.length})
            </h3>
            <div className="flex items-center space-x-4">
              <select
                value={filterLevel}
                onChange={(e) => setFilterLevel(e.target.value)}
                className="input-field w-40"
              >
                <option value="all">All Levels</option>
                {Array.from(new Set(rankings?.map(r => r.level) || [])).map(level => (
                  <option key={level} value={level}>{level}</option>
                ))}
              </select>
              <button
                onClick={() => {
                  const response = fetch('/api/founders/rankings/export')
                  response.then(r => r.blob()).then(blob => {
                    const url = window.URL.createObjectURL(blob)
                    const a = document.createElement('a')
                    a.href = url
                    a.download = `rankings_${new Date().toISOString().split('T')[0]}.csv`
                    a.click()
                  })
                }}
                className="btn-secondary flex items-center"
              >
                <Download className="h-4 w-4 mr-2" />
                Export
              </button>
            </div>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-secondary-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 uppercase">Founder</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 uppercase">Level</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 uppercase">Confidence</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 uppercase">Reasoning</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 uppercase">Evidence</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-secondary-200">
              {filteredRankings.map((ranking) => (
                <tr key={ranking.id} className="hover:bg-secondary-50">
                  <td className="px-6 py-4">
                    <div>
                      <div className="font-medium text-secondary-900">{ranking.name}</div>
                      <div className="text-sm text-secondary-500">{ranking.company}</div>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${levelColors[ranking.level] || 'bg-gray-100 text-gray-800'}`}>
                      {ranking.level}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex items-center">
                      <div className="w-16 bg-secondary-200 rounded-full h-2 mr-3">
                        <div 
                          className="bg-primary-600 h-2 rounded-full"
                          style={{ width: `${ranking.confidenceScore * 100}%` }}
                        />
                      </div>
                      <span className="text-sm text-secondary-900">
                        {(ranking.confidenceScore * 100).toFixed(0)}%
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 max-w-xs">
                    <div className="text-sm text-secondary-900 truncate" title={ranking.reasoning}>
                      {ranking.reasoning}
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <div className="text-sm text-secondary-900">
                      {ranking.evidence.length} items
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

export default FounderRanking
