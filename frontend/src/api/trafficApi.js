import axios from 'axios';

const BASE = '/api';

export const fetchStatus  = ()       => axios.get(`${BASE}/status`);
export const fetchSensors = ()       => axios.get(`${BASE}/sensors`);
export const fetchHistory = (id)     => axios.get(`${BASE}/history/${id}`);
export const fetchMetrics = ()       => axios.get(`${BASE}/metrics`);
export const fetchAlerts  = ()       => axios.get(`${BASE}/alerts`);
export const fetchEval    = ()       => axios.get(`${BASE}/evaluation`);

// [F2] Wire frontend settings to backend query params
export const fetchTraffic = (settings = {}) => {
  const params = {
    pCong:       settings.pCongThreshold      ?? 0.6,
    uncertainty: settings.uncertaintyCeil     ?? undefined,
    eta:         settings.propagationThreshold ?? 0.5,
  };
  // Only send uncertainty if user has explicitly set it
  if (!settings.uncertaintyOverride) delete params.uncertainty;
  return axios.get(`${BASE}/traffic`, { params });
};
