export interface PipelineStatus {
  step: string
  status: 'pending' | 'running' | 'completed' | 'error'
  message?: string
}

export interface PipelineResults {
  companies: any[]
  founders: any[]
  jobId: string
}

export interface Company {
  id: string
  name: string
  sector: string
  founded_year: string | number
  ai_focus: string
}

export interface MarketAnalysisData {
  company_name: string
  sector: string
  founded_year: number
  market_size_billion: number
  cagr_percent: number
  timing_score: number
  us_sentiment: number
  sea_sentiment: number
  competitor_count: number
  total_funding_billion: number
  momentum_score: number
  market_stage: string
  confidence_score: number
  analysis_date: string
  execution_time: number
  
  // Comprehensive text analysis
  market_overview?: string
  market_size_analysis?: string
  growth_drivers?: string
  timing_analysis?: string
  regional_analysis?: string
  competitive_landscape?: string
  investment_climate?: string
  regulatory_environment?: string
  technology_trends?: string
  consumer_adoption?: string
  supply_chain_analysis?: string
  risk_assessment?: string
  strategic_recommendations?: string
  
  // Structured insights
  key_trends?: string[]
  major_players?: string[]
  barriers_to_entry?: string[]
  opportunities?: string[]
  threats?: string[]
  regulatory_changes?: string[]
  emerging_technologies?: string[]
}

export interface CheckpointInfo {
  id: string
  job_id?: string
  stage?: string
  created_at: string
  foundation_year: number
  latest_stage: string
  completion_percentage: number
  stages_completed: number
}
