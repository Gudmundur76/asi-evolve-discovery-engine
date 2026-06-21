export interface Discovery {
  id: number;
  candidate_id: string;
  created_at: string;
  target_chembl_id: string;
  target_name: string;
  smiles_hint: string;
  predicted_affinity_nm: number;
  training_best_nm: number;
  improvement_factor: number;
  docking_score: number | null;
  docking_passed: boolean;
  admet_druglikeness_score: number | null;
  admet_passed: boolean;
  confidence_score: number;
  overall_passed: boolean;
  evidence_pdf_path: string | null;
  cycle_number: number;
}

export interface LoopStatus {
  running: boolean;
  cycle_count: number;
  current_best_affinity: number;
  last_cycle_time: string;
  next_cycle_time: string;
  target_chembl_id: string;
  target_name: string;
}

export interface Candidate {
  cycle_id: number;
  timestamp: string;
  new_smiles: string;
  predicted_affinity_nm: number;
  is_best_so_far: boolean;
  lesson: string;
}

export interface CycleResult {
  cycle_id: number;
  predicted_affinity_nm: number;
  improvement: number;
  is_best_so_far: boolean;
  lesson: string;
  fingerprint_diff: number[];
}

export interface WebSocketMessage {
  type: 'loop_status' | 'new_discovery' | 'cycle_complete' | 'error';
  payload: unknown;
}

export interface AdmetProperty {
  name: string;
  value: number;
  threshold: number;
  unit: string;
  passed: boolean;
}

export interface DiscoveryFilters {
  dateFrom: string;
  dateTo: string;
  minConfidence: number;
  sortBy: 'affinity' | 'confidence' | 'date';
  sortOrder: 'asc' | 'desc';
}
