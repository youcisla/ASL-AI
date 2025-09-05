import React from 'react';
import UploadCard from '../components/UploadCard';

const Home: React.FC = () => {
  return (
    <main style={{ minHeight: '80vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-start', padding: '2rem 0' }}>
      {/* Header */}
      <section style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
        <h1 style={{ fontWeight: 800, fontSize: '2.5rem', color: 'var(--accent)' }}>ASL Hand Sign Classifier</h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '1.2rem', maxWidth: 600, margin: '1rem auto 0 auto' }}>
          Upload an image of an ASL hand sign to get instant predictions using our advanced neural network.
        </p>
      </section>

      {/* Upload Section */}
      <section style={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', marginBottom: '2.5rem' }}>
        <UploadCard />
      </section>

      {/* How it works Section */}
      <section style={{ width: '100%', maxWidth: 900, margin: '0 auto', marginTop: '2rem' }}>
        <div className="card" style={{ borderRadius: 'var(--border-radius)', boxShadow: 'var(--shadow)', background: 'var(--card-bg)', padding: '2rem' }}>
          <h2 style={{ color: 'var(--accent)', fontWeight: 700, textAlign: 'center', marginBottom: '2rem' }}>How it works</h2>
          <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap', gap: '2rem' }}>
            <div style={{ flex: 1, minWidth: 220, textAlign: 'center' }}>
              <div style={{ marginBottom: 16 }}>
                <svg width="48" height="48" fill="currentColor" className="bi bi-cloud-upload" viewBox="0 0 16 16" style={{ color: 'var(--accent)' }}>
                  <path fillRule="evenodd" d="M4.406 1.342A5.53 5.53 0 0 1 8 0c2.69 0 4.923 2 5.166 4.579C14.758 4.804 16 6.137 16 7.773 16 9.569 14.502 11 12.687 11H10a.5.5 0 0 1 0-1h2.688C13.979 10 15 8.988 15 7.773c0-1.216-1.02-2.228-2.313-2.228h-.5v-.5C12.188 2.825 10.328 1 8 1a4.53 4.53 0 0 0-3.941 2.341z"/>
                  <path fillRule="evenodd" d="M7.646 4.146a.5.5 0 0 1 .708 0l3 3a.5.5 0 0 1-.708.708L8.5 5.707V14.5a.5.5 0 0 1-1 0V5.707L5.354 7.854a.5.5 0 1 1-.708-.708l3-3z"/>
                </svg>
              </div>
              <h4 style={{ color: 'var(--accent)', fontWeight: 700 }}>1. Upload</h4>
              <p style={{ color: 'var(--text-secondary)' }}>Upload or drag & drop an image of an ASL hand sign</p>
            </div>
            <div style={{ flex: 1, minWidth: 220, textAlign: 'center' }}>
              <div style={{ marginBottom: 16 }}>
                <svg width="48" height="48" fill="currentColor" className="bi bi-cpu" viewBox="0 0 16 16" style={{ color: 'var(--accent2)' }}>
                  <path d="M5 0a.5.5 0 0 1 .5.5V2h1V.5a.5.5 0 0 1 1 0V2h1V.5a.5.5 0 0 1 1 0V2h1V.5a.5.5 0 0 1 1 0V2A2.5 2.5 0 0 1 14 4.5h1.5a.5.5 0 0 1 0 1H14v1h1.5a.5.5 0 0 1 0 1H14v1h1.5a.5.5 0 0 1 0 1H14v1h1.5a.5.5 0 0 1 0 1H14a2.5 2.5 0 0 1-2.5 2.5v1.5a.5.5 0 0 1-1 0V14h-1v1.5a.5.5 0 0 1-1 0V14h-1v1.5a.5.5 0 0 1-1 0V14h-1v1.5a.5.5 0 0 1-1 0V14A2.5 2.5 0 0 1 2 11.5H.5a.5.5 0 0 1 0-1H2v-1H.5a.5.5 0 0 1 0-1H2v-1H.5a.5.5 0 0 1 0-1H2v-1H.5a.5.5 0 0 1 0-1H2A2.5 2.5 0 0 1 4.5 2V.5A.5.5 0 0 1 5 0zm-.5 3A1.5 1.5 0 0 0 3 4.5v7A1.5 1.5 0 0 0 4.5 13h7a1.5 1.5 0 0 0 1.5-1.5v-7A1.5 1.5 0 0 0 11.5 3h-7zM5 6.5A1.5 1.5 0 0 1 6.5 5h3A1.5 1.5 0 0 1 11 6.5v3A1.5 1.5 0 0 1 9.5 11h-3A1.5 1.5 0 0 1 5 9.5v-3zM6.5 6a.5.5 0 0 0-.5.5v3a.5.5 0 0 0 .5.5h3a.5.5 0 0 0 .5-.5v-3a.5.5 0 0 0-.5-.5h-3z"/>
                </svg>
              </div>
              <h4 style={{ color: 'var(--accent2)', fontWeight: 700 }}>2. Analyze</h4>
              <p style={{ color: 'var(--text-secondary)' }}>Our model processes the image and identifies the sign</p>
            </div>
            <div style={{ flex: 1, minWidth: 220, textAlign: 'center' }}>
              <div style={{ marginBottom: 16 }}>
                <svg width="48" height="48" fill="currentColor" className="bi bi-graph-up" viewBox="0 0 16 16" style={{ color: 'var(--accent3)' }}>
                  <path fillRule="evenodd" d="M0 0h1v15h15v1H0V0Zm14.817 3.113a.5.5 0 0 1 .07.704l-4.5 5.5a.5.5 0 0 1-.74.037L7.06 6.767l-3.656 5.027a.5.5 0 0 1-.808-.588l4-5.5a.5.5 0 0 1 .758-.06l2.609 2.61 4.15-5.073a.5.5 0 0 1 .704-.07Z"/>
                </svg>
              </div>
              <h4 style={{ color: 'var(--accent3)', fontWeight: 700 }}>3. Results</h4>
              <p style={{ color: 'var(--text-secondary)' }}>Get top 3 predictions with confidence scores</p>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
};

export default Home;
