import React, { useState } from 'react';
import TopBar            from './components/TopBar';
import StatBar           from './components/StatBar';
import MethodologyBar    from './components/MethodologyBar';
import SensorCard        from './components/SensorCard';
import Charts            from './components/Charts';
import AlertLog          from './components/AlertLog';
import Sidebar           from './components/Sidebar';
import NetworkGraph      from './components/NetworkGraph';
import MetricsPanel      from './components/MetricsPanel';
import SensorDetailModal from './components/SensorDetailModal';
import SettingsPanel     from './components/SettingsPanel';
import useTrafficPoll    from './hooks/useTrafficPoll';

const DEFAULT_SETTINGS = {
  pCongThreshold:       0.6,
  uncertaintyCeil:      0.05,
  uncertaintyOverride:  false,   // [F5] auto by default
  propagationThreshold: 0.5,
  pollInterval:         3000,
  showNetworkGraph:     true,
  showMetricsPanel:     true,
};

export default function App() {
  const [settings,     setSettings]     = useState(DEFAULT_SETTINGS);
  const [showSettings, setShowSettings] = useState(false);
  const [modalSensor,  setModalSensor]  = useState(null);

  const {
    sensors, summary, alerts, online, loading,
    selectedId, selectSensor,
    speedHist, anomHist, warnHist, congHist,
    chartTick, stepCount,
  } = useTrafficPoll(settings);

  /* ── Loading screen ────────────────────────────────── */
  if (loading) return (
    <div style={{ display:'flex', flexDirection:'column', alignItems:'center',
      justifyContent:'center', height:'100vh', gap:16, background:'#0d1424' }}>
      <div style={{ width:40, height:40, border:'3px solid #1e2d47',
        borderTop:'3px solid #3b82f6', borderRadius:'50%', animation:'spin 1s linear infinite' }}/>
      <div style={{ color:'#8898b3', fontSize:13 }}>Connecting to AAR-EWS backend…</div>
      <div style={{ color:'#344055', fontSize:11 }}>
        Make sure Flask is running: <code style={{ color:'#3b82f6' }}>cd backend && python app.py</code>
      </div>
      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
    </div>
  );

  /* ── Offline screen ────────────────────────────────── */
  if (!online) return (
    <div style={{ display:'flex', flexDirection:'column', alignItems:'center',
      justifyContent:'center', height:'100vh', gap:14, textAlign:'center',
      padding:24, background:'#0d1424' }}>
      <div style={{ fontSize:32 }}>⚠️</div>
      <div style={{ fontSize:17, fontWeight:700 }}>Backend offline</div>
      <div style={{ color:'#8898b3', fontSize:13, maxWidth:440, lineHeight:1.7 }}>
        Start the Flask API, then refresh:
      </div>
      <code style={{ background:'#131929', border:'1px solid #1e2d47', borderRadius:8,
        padding:'10px 18px', fontSize:13, color:'#3b82f6' }}>
        cd backend &amp;&amp; python generate_dataset.py &amp;&amp; python app.py
      </code>
      <div style={{ color:'#344055', fontSize:11, marginTop:6 }}>
        First run trains the TCN on the bundled synthetic METR-LA dataset (~60 s).
      </div>
    </div>
  );

  /* ── Main layout ───────────────────────────────────── */
  return (
    <div style={{ display:'grid', gridTemplateRows:'54px 1fr', height:'100vh', background:'#0d1424' }}>
      <TopBar online={online} summary={summary} stepCount={stepCount}/>

      <div style={{ display:'grid', gridTemplateColumns:'220px 1fr', overflow:'hidden' }}>
        <Sidebar sensors={sensors} selectedId={selectedId} onSelect={selectSensor}/>

        <main style={{ overflowY:'auto', padding:'16px 18px' }}>
          {/* Settings button row */}
          <div style={{ display:'flex', justifyContent:'flex-end', alignItems:'center',
            gap:12, marginBottom:14 }}>
            <div style={{ fontSize:11, color:'#344055' }}>
              step {stepCount} · poll {settings.pollInterval/1000}s ·
              η={settings.propagationThreshold} · θ_C={settings.pCongThreshold}
            </div>
            <button onClick={() => setShowSettings(true)} style={{
              background:'#131929', border:'1px solid #1e2d47', borderRadius:8,
              color:'#8898b3', cursor:'pointer', padding:'6px 14px', fontSize:12,
            }}>⚙ Settings</button>
          </div>

          <StatBar summary={summary}/>
          <MethodologyBar/>
          {settings.showMetricsPanel && <MetricsPanel/>}
          {settings.showNetworkGraph  && (
            <NetworkGraph sensors={sensors} selectedId={selectedId} onSelect={selectSensor}/>
          )}

          {/* Sensor cards grid */}
          <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(260px,1fr))',
            gap:10, marginBottom:18 }}>
            {sensors.map(s => (
              <SensorCard key={s.sensor_id} sensor={s} selected={s.sensor_id===selectedId}
                onClick={() => { selectSensor(s.sensor_id); setModalSensor(s); }}/>
            ))}
          </div>

          <Charts
            speedHist={speedHist} anomHist={anomHist}
            warnHist={warnHist}   congHist={congHist}
            sensors={sensors}     chartTick={chartTick}/>

          <AlertLog alerts={alerts}/>

          <div style={{ fontSize:10, color:'#1e2d47', textAlign:'center', paddingBottom:20 }}>
            AAR-EWS · G. Pullaiah College of Engineering and Technology ·
            Synthetic METR-LA dataset · React 18 + Flask + PyTorch N-TCN
          </div>
        </main>
      </div>

      {modalSensor && (
        <SensorDetailModal
          sensor={sensors.find(s => s.sensor_id===modalSensor.sensor_id) || modalSensor}
          onClose={() => setModalSensor(null)}/>
      )}
      {showSettings && (
        <SettingsPanel
          settings={settings}
          onChange={setSettings}
          onClose={() => setShowSettings(false)}/>
      )}
    </div>
  );
}
