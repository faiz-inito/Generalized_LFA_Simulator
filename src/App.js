import { Routes, Route } from "react-router-dom";
import LFASimulator from "./LFASimulator";
import CompetitiveLFASimulator from "./CompetitiveLFASimulator";
import Home from "./Home";
import GeneralizedLFASimulator from "./GeneralizedLFASimulator";

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/sandwich-lfa" element={<LFASimulator />} />
      <Route path="/competitive-lfa" element={<CompetitiveLFASimulator />} />
      <Route path="/generalized-lfa" element={<GeneralizedLFASimulator />} />
    </Routes>
  );
}

export default App;