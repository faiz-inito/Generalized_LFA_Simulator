import { useNavigate } from "react-router-dom";

const T = {
  bg: "#070a08",
  card: "#0f1510",
  surface: "#0b100d",
  border: "#1a2820",
  borderHi: "#2a4535",
  text: "#dde4f0",
  muted: "#4a5a50",
  accent: "#fbbf24",
};

function Home() {
  const navigate = useNavigate();

  const Card = ({ title, desc, path, gradient }) => (
    <div
      onClick={() => navigate(path)}
      style={{
        flex: 1,
        minWidth: 260,
        background: T.card,
        border: `1px solid ${T.border}`,
        borderRadius: 14,
        padding: "22px 20px",
        cursor: "pointer",
        transition: "all 0.25s ease",
        position: "relative",
        overflow: "hidden",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.transform = "translateY(-4px)";
        e.currentTarget.style.borderColor = T.borderHi;
        e.currentTarget.style.boxShadow =
          "0 10px 30px rgba(0,0,0,0.4)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = "translateY(0)";
        e.currentTarget.style.borderColor = T.border;
        e.currentTarget.style.boxShadow = "none";
      }}
    >
      {/* subtle gradient glow */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: gradient,
          opacity: 0.08,
        }}
      />

      <div style={{ position: "relative", zIndex: 1 }}>
        <h2
          style={{
            fontFamily: "'Syne', sans-serif",
            fontWeight: 800,
            fontSize: 20,
            marginBottom: 6,
            background: gradient,
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}
        >
          {title}
        </h2>

        <p
          style={{
            fontSize: 12,
            color: T.muted,
            lineHeight: 1.6,
          }}
        >
          {desc}
        </p>
      </div>
    </div>
  );

  return (
    <div
      style={{
        minHeight: "100vh",
        background: T.bg,
        color: T.text,
        fontFamily: "'DM Mono', monospace",
        padding: "40px 20px",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
      }}
    >
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@700;800&display=swap');
      `}</style>

      {/* HEADER */}
      <div style={{ textAlign: "center", marginBottom: 40 }}>
        <div
          style={{
            fontSize: 10,
            letterSpacing: 4,
            color: T.muted,
            marginBottom: 6,
          }}
        >
          LATERAL FLOW ASSAY PLATFORM
        </div>

        <h1
          style={{
            fontFamily: "'Syne', sans-serif",
            fontSize: 32,
            fontWeight: 800,
            letterSpacing: -0.5,
            background: "linear-gradient(110deg,#fbbf24,#4ade80)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}
        >
          Simulation Suite
        </h1>

        <p
          style={{
            marginTop: 10,
            fontSize: 13,
            color: T.muted,
          }}
        >
          Explore sandwich and competitive LFA mechanisms with physics-based models
        </p>
      </div>

      {/* CARDS */}
      <div
        style={{
          display: "flex",
          gap: 18,
          width: "100%",
          maxWidth: 720,
          flexWrap: "wrap",
        }}
      >
        <Card
          title="Sandwich LFA Simulator"
          desc="Classic assay where signal increases with analyte concentration. Visualize binding kinetics, flow, and signal formation."
          path="/sandwich-lfa"
          gradient="linear-gradient(110deg,#38bdf8,#60a5fa)"
        />

        <Card
          title="Competitive LFA Simulator"
          desc="Inverted signal assay. Higher analyte suppresses test line intensity. Analyze competition dynamics and T/C thresholds."
          path="/competitive-lfa"
          gradient="linear-gradient(110deg,#fbbf24,#4ade80)"
        />

        <Card
          title="Generalized LFA Simulator"
          desc={
            <>
              Generalized LFA provides complete control over assay design, including the number and type of lines, the number of analytes in the sample, the number of conjugate antibodies, and virtually every other key parameter.
            </>
          }
          path="/generalized-lfa"
          gradient="linear-gradient(110deg,#fbbf24,#4ade80)"
        />
      </div>
    </div>
  );
}

export default Home;