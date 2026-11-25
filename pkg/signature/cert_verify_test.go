//
// Copyright 2026 The Sigstore Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package signature

import (
	"bytes"
	"context"
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha256"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"math/big"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Fixed reference instants. No test in this file depends on time.Now.
var (
	baseTime       = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	signingInstant = baseTime.Add(-1 * time.Hour)
)

type certHierarchy struct {
	root          *x509.Certificate
	rootKey       *ecdsa.PrivateKey
	intermediates []*x509.Certificate
	intermediateK []*ecdsa.PrivateKey
	leaf          *x509.Certificate
	leafKey       *ecdsa.PrivateKey
}

type timeWindow struct {
	notBefore time.Time
	notAfter  time.Time
}

// tHelper is the subset of testing.T/testing.F used by the fixture helpers.
type tHelper interface {
	Helper()
	Fatal(args ...any)
	Fatalf(format string, args ...any)
	Error(args ...any)
	Errorf(format string, args ...any)
}

func mustCreateCert(t tHelper, template, parent *x509.Certificate, pub crypto.PublicKey, signer crypto.Signer) *x509.Certificate {
	t.Helper()
	der, err := x509.CreateCertificate(rand.Reader, template, parent, pub, signer)
	if err != nil {
		t.Fatalf("creating certificate: %v", err)
	}
	cert, err := x509.ParseCertificate(der)
	if err != nil {
		t.Fatalf("parsing created certificate: %v", err)
	}
	return cert
}

func mustGenerateECDSAKey(t tHelper) *ecdsa.PrivateKey {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("generating ECDSA key: %v", err)
	}
	return key
}

// buildHierarchy creates a root, nIntermediates intermediate CAs and a leaf
// with deterministic validity windows around baseTime. Optional windows
// override the default leaf (first) and intermediate (second) windows.
func buildHierarchy(t tHelper, nIntermediates int, windows ...timeWindow) *certHierarchy {
	t.Helper()
	leafWindow := timeWindow{notBefore: baseTime.Add(-time.Hour), notAfter: baseTime.Add(24 * time.Hour)}
	intermediateWindow := timeWindow{notBefore: baseTime.Add(-2 * time.Hour), notAfter: baseTime.Add(48 * time.Hour)}
	if len(windows) > 0 {
		leafWindow = windows[0]
	}
	if len(windows) > 1 {
		intermediateWindow = windows[1]
	}

	h := &certHierarchy{}
	h.rootKey = mustGenerateECDSAKey(t)
	rootTemplate := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "test-root", Organization: []string{"sigstore-test"}},
		NotBefore:             baseTime.Add(-24 * time.Hour),
		NotAfter:              baseTime.Add(72 * time.Hour),
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
		BasicConstraintsValid: true,
		IsCA:                  true,
	}
	h.root = mustCreateCert(t, rootTemplate, rootTemplate, &h.rootKey.PublicKey, h.rootKey)

	// Build intermediates root-side down: root -> C0 -> C1 -> ... -> Cn-1,
	// then store them in reverse so index 0 is the CA closest to the leaf:
	// root -> I[n-1] -> ... -> I0 -> leaf. The supplied intermediate list is
	// therefore [I0, I1, ...], with the first entry directly issuing the leaf,
	// matching the ordering required by the verifier.
	certs := make([]*x509.Certificate, nIntermediates)
	keys := make([]*ecdsa.PrivateKey, nIntermediates)
	parent := h.root
	parentKey := h.rootKey
	for i := 0; i < nIntermediates; i++ {
		key := mustGenerateECDSAKey(t)
		template := &x509.Certificate{
			SerialNumber:          big.NewInt(int64(i + 2)),
			Subject:               pkix.Name{CommonName: fmt.Sprintf("test-intermediate-%d", nIntermediates-1-i)},
			NotBefore:             intermediateWindow.notBefore,
			NotAfter:              intermediateWindow.notAfter,
			KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
			ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageCodeSigning},
			BasicConstraintsValid: true,
			IsCA:                  true,
		}
		certs[i] = mustCreateCert(t, template, parent, &key.PublicKey, parentKey)
		keys[i] = key
		parent = certs[i]
		parentKey = key
	}
	for i := range certs {
		h.intermediates = append(h.intermediates, certs[nIntermediates-1-i])
		h.intermediateK = append(h.intermediateK, keys[nIntermediates-1-i])
	}

	h.leafKey = mustGenerateECDSAKey(t)
	leafParent := h.root
	leafParentKey := h.rootKey
	if nIntermediates > 0 {
		leafParent = h.intermediates[0]
		leafParentKey = h.intermediateK[0]
	}
	leafTemplate := &x509.Certificate{
		SerialNumber: big.NewInt(100),
		Subject:      pkix.Name{CommonName: "test-leaf"},
		NotBefore:    leafWindow.notBefore,
		NotAfter:     leafWindow.notAfter,
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageCodeSigning},
		IsCA:         false,
	}
	h.leaf = mustCreateCert(t, leafTemplate, leafParent, &h.leafKey.PublicKey, leafParentKey)
	return h
}

// buildLeafWithEKU re-issues the leaf of h with the supplied EKUs.
func buildLeafWithEKU(t tHelper, h *certHierarchy, ekus []x509.ExtKeyUsage) *x509.Certificate {
	t.Helper()
	parent := h.root
	parentKey := h.rootKey
	if len(h.intermediates) > 0 {
		parent = h.intermediates[0]
		parentKey = h.intermediateK[0]
	}
	template := &x509.Certificate{
		SerialNumber: big.NewInt(101),
		Subject:      pkix.Name{CommonName: "test-leaf-eku"},
		NotBefore:    baseTime.Add(-time.Hour),
		NotAfter:     baseTime.Add(24 * time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  ekus,
		IsCA:         false,
	}
	return mustCreateCert(t, template, parent, &h.leafKey.PublicKey, parentKey)
}

// buildLeafAsCA re-issues the leaf of h with the CA bit set.
func buildLeafAsCA(t tHelper, h *certHierarchy) *x509.Certificate {
	t.Helper()
	parent := h.root
	parentKey := h.rootKey
	if len(h.intermediates) > 0 {
		parent = h.intermediates[0]
		parentKey = h.intermediateK[0]
	}
	template := &x509.Certificate{
		SerialNumber:          big.NewInt(102),
		Subject:               pkix.Name{CommonName: "test-leaf-ca"},
		NotBefore:             baseTime.Add(-time.Hour),
		NotAfter:              baseTime.Add(24 * time.Hour),
		KeyUsage:              x509.KeyUsageCertSign,
		BasicConstraintsValid: true,
		IsCA:                  true,
	}
	return mustCreateCert(t, template, parent, &h.leafKey.PublicKey, parentKey)
}

func mustPEM(t tHelper, cert *x509.Certificate) []byte {
	t.Helper()
	out, err := encodeCertPEM(cert)
	if err != nil {
		t.Fatalf("encoding certificate: %v", err)
	}
	return out
}

func encodeCertPEM(cert *x509.Certificate) ([]byte, error) {
	if cert == nil {
		return nil, errors.New("nil certificate")
	}
	return pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: cert.Raw}), nil
}

func mustConcatPEM(t tHelper, certs ...*x509.Certificate) []byte {
	t.Helper()
	var buf bytes.Buffer
	for _, cert := range certs {
		buf.Write(mustPEM(t, cert))
	}
	return buf.Bytes()
}

func rootPool(certs ...*x509.Certificate) *x509.CertPool {
	pool := x509.NewCertPool()
	for _, cert := range certs {
		pool.AddCert(cert)
	}
	return pool
}

func signMessage(t tHelper, key *ecdsa.PrivateKey, message []byte) []byte {
	t.Helper()
	digest := sha256.Sum256(message)
	sig, err := ecdsa.SignASN1(rand.Reader, key, digest[:])
	if err != nil {
		t.Fatalf("signing message: %v", err)
	}
	return sig
}

func defaultPolicy(h *certHierarchy) ChainPolicy {
	return ChainPolicy{
		Roots:     rootPool(h.root),
		KeyUsages: []x509.ExtKeyUsage{x509.ExtKeyUsageCodeSigning},
	}
}

func mustVerifier(t tHelper, policy ChainPolicy) *CertificateVerifier {
	t.Helper()
	verifier, err := NewCertificateVerifier(policy)
	if err != nil {
		t.Fatalf("NewCertificateVerifier: %v", err)
	}
	return verifier
}

// countingAuthority records call counts and the requests it receives. It is
// safe for concurrent use. When block is set, every call parks until the
// context is done.
type countingAuthority struct {
	calls    atomic.Int64
	mu       sync.Mutex
	last     SigningTimeRequest
	provide  func() (time.Time, error)
	block    bool
	entered  chan struct{}
	entered0 sync.Once
}

func newCountingAuthority(provide func() (time.Time, error)) *countingAuthority {
	return &countingAuthority{provide: provide, entered: make(chan struct{})}
}

func (a *countingAuthority) authority(ctx context.Context, request SigningTimeRequest) (time.Time, error) {
	a.calls.Add(1)
	a.entered0.Do(func() { close(a.entered) })
	a.mu.Lock()
	a.last = request
	a.mu.Unlock()
	if a.block {
		<-ctx.Done()
		return time.Time{}, ctx.Err()
	}
	if a.provide == nil {
		return time.Time{}, errors.New("no signing time configured")
	}
	return a.provide()
}

func (a *countingAuthority) lastRequest() SigningTimeRequest {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.last
}

// countingReader verifies that Verify consumes the full message stream.
type countingReader struct {
	r     io.Reader
	count int64
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.count += int64(n)
	return n, err
}

func assertErrorIs(t tHelper, err error, target error) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("expected error matching %v, got %v", target, err)
	}
}

func assertValidityError(t tHelper, err error, role CertRole, expired bool, effective time.Time) *CertificateValidityError {
	t.Helper()
	var validityErr *CertificateValidityError
	if !errors.As(err, &validityErr) {
		t.Fatalf("expected *CertificateValidityError, got %v", err)
	}
	if validityErr.Role != role {
		t.Errorf("expected role %q, got %q", role, validityErr.Role)
	}
	if validityErr.Expired != expired {
		t.Errorf("expected Expired=%v, got %v", expired, validityErr.Expired)
	}
	if !validityErr.EffectiveTime.Equal(effective.UTC()) {
		t.Errorf("expected effective time %v, got %v", effective.UTC(), validityErr.EffectiveTime)
	}
	if validityErr.Subject == "" {
		t.Errorf("expected non-empty subject")
	}
	if validityErr.NotBefore.Location() != time.UTC || validityErr.NotAfter.Location() != time.UTC ||
		validityErr.EffectiveTime.Location() != time.UTC {
		t.Errorf("validity error times must be UTC: %+v", validityErr)
	}
	// The RFC3339 rendering of the effective time must always appear; the
	// boundary relevant to the failure mode must appear too.
	if !strings.Contains(validityErr.Error(), validityErr.EffectiveTime.Format(time.RFC3339)) {
		t.Errorf("error text must contain the UTC RFC3339 effective time, got %q", validityErr.Error())
	}
	boundary := validityErr.NotAfter
	if !expired {
		boundary = validityErr.NotBefore
	}
	if !strings.Contains(validityErr.Error(), boundary.Format(time.RFC3339)) {
		t.Errorf("error text must contain the relevant UTC RFC3339 boundary, got %q", validityErr.Error())
	}
	return validityErr
}

func TestCertificateVerifierNewPolicy(t *testing.T) {
	h := buildHierarchy(t, 1)
	base := defaultPolicy(h)

	tests := []struct {
		name   string
		policy ChainPolicy
	}{
		{"nil roots", ChainPolicy{KeyUsages: base.KeyUsages}},
		{"empty roots", ChainPolicy{Roots: x509.NewCertPool(), KeyUsages: base.KeyUsages}},
		{"empty key usages", ChainPolicy{Roots: base.Roots}},
		{"unsupported hash", ChainPolicy{Roots: base.Roots, KeyUsages: base.KeyUsages, Hash: crypto.MD5}},
		{"unavailable hash", ChainPolicy{Roots: base.Roots, KeyUsages: base.KeyUsages, Hash: crypto.BLAKE2s_256}},
		{"max intermediates zero-range", ChainPolicy{Roots: base.Roots, KeyUsages: base.KeyUsages, MaxIntermediates: -1}},
		{"max intermediates too large", ChainPolicy{Roots: base.Roots, KeyUsages: base.KeyUsages, MaxIntermediates: 17}},
		{"negative clock skew", ChainPolicy{Roots: base.Roots, KeyUsages: base.KeyUsages, ClockSkew: -time.Nanosecond}},
		{"clock skew too large", ChainPolicy{Roots: base.Roots, KeyUsages: base.KeyUsages, ClockSkew: 5*time.Minute + time.Nanosecond}},
		{"mode out of range", ChainPolicy{Roots: base.Roots, KeyUsages: base.KeyUsages, Mode: TimeMode(99)}},
		{"signing mode without authority", ChainPolicy{Roots: base.Roots, KeyUsages: base.KeyUsages, Mode: TimeModeSigning}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			verifier, err := NewCertificateVerifier(tt.policy)
			if verifier != nil {
				t.Errorf("expected nil verifier")
			}
			assertErrorIs(t, err, ErrInvalidPolicy)
		})
	}

	t.Run("defaults and allowed hashes", func(t *testing.T) {
		for _, hash := range []crypto.Hash{0, crypto.SHA256, crypto.SHA384, crypto.SHA512} {
			policy := defaultPolicy(h)
			policy.Hash = hash
			v := mustVerifier(t, policy)
			if v.hash != crypto.SHA256 && hash != 0 {
				if v.hash != hash {
					t.Errorf("expected hash %v, got %v", hash, v.hash)
				}
			}
			if v.maxIntermediates != defaultMaxIntermediateCerts {
				t.Errorf("expected default max intermediates %d, got %d", defaultMaxIntermediateCerts, v.maxIntermediates)
			}
		}
	})

	t.Run("bounds accepted", func(t *testing.T) {
		policy := defaultPolicy(h)
		policy.MaxIntermediates = 1
		policy.ClockSkew = 5 * time.Minute
		mustVerifier(t, policy)
		policy.MaxIntermediates = 16
		mustVerifier(t, policy)
	})
}

func TestCertificateVerifierSuccessCurrentMode(t *testing.T) {
	h := buildHierarchy(t, 1)
	verifier := mustVerifier(t, defaultPolicy(h))
	message := []byte("hello certificate chain")
	signature := signMessage(t, h.leafKey, message)

	result, err := verifier.Verify(
		context.Background(),
		mustPEM(t, h.leaf),
		mustPEM(t, h.intermediates[0]),
		bytes.NewReader(message),
		bytes.NewReader(signature),
		baseTime,
	)
	if err != nil {
		t.Fatalf("Verify failed: %v", err)
	}
	if result.Leaf == nil || !result.Leaf.Equal(h.leaf) {
		t.Fatalf("unexpected leaf in result")
	}
	if len(result.Chain) != 3 {
		t.Fatalf("expected chain of length 3, got %d", len(result.Chain))
	}
	expectedRoles := []CertRole{RoleLeaf, RoleIntermediate, RoleRoot}
	if len(result.Roles) != len(expectedRoles) {
		t.Fatalf("expected %d roles, got %d", len(expectedRoles), len(result.Roles))
	}
	for i, role := range expectedRoles {
		if result.Roles[i] != role {
			t.Errorf("role %d: expected %q, got %q", i, role, result.Roles[i])
		}
	}
	if !result.Chain[0].Equal(h.leaf) || !result.Chain[1].Equal(h.intermediates[0]) || !result.Chain[2].Equal(h.root) {
		t.Errorf("chain order is not leaf -> intermediate -> root")
	}
	if result.Mode != TimeModeCurrent {
		t.Errorf("expected current mode")
	}
	if !result.EffectiveTime.Equal(baseTime) {
		t.Errorf("expected effective time %v, got %v", baseTime, result.EffectiveTime)
	}
	if result.EffectiveTime.Location() != time.UTC {
		t.Errorf("effective time must be UTC")
	}
	if result.SigningTime != nil {
		t.Errorf("SigningTime must be nil in current mode")
	}

	// Returned slices must not share storage with internal state.
	result.Chain[0] = nil
	result2, err := verifier.Verify(
		context.Background(), mustPEM(t, h.leaf), mustPEM(t, h.intermediates[0]),
		bytes.NewReader(message), bytes.NewReader(signature), baseTime)
	if err != nil || result2.Chain[0] == nil {
		t.Fatalf("verifier state leaked through result slice")
	}
}

func TestCertificateVerifierDirectRoot(t *testing.T) {
	h := buildHierarchy(t, 0)
	verifier := mustVerifier(t, defaultPolicy(h))
	message := []byte("no intermediates")
	signature := signMessage(t, h.leafKey, message)

	result, err := verifier.Verify(context.Background(), mustPEM(t, h.leaf), nil,
		bytes.NewReader(message), bytes.NewReader(signature), baseTime)
	if err != nil {
		t.Fatalf("Verify failed: %v", err)
	}
	if len(result.Chain) != 2 || result.Roles[0] != RoleLeaf || result.Roles[1] != RoleRoot {
		t.Fatalf("unexpected chain/roles: %v", result.Roles)
	}
}

func TestCertificateVerifierValidity(t *testing.T) {
	t.Run("leaf expired", func(t *testing.T) {
		h := buildHierarchy(t, 1, timeWindow{notBefore: baseTime.Add(-2 * time.Hour), notAfter: baseTime.Add(-time.Hour)})
		verifier := mustVerifier(t, defaultPolicy(h))
		message := []byte("expired leaf")
		signature := signMessage(t, h.leafKey, message)
		_, err := verifier.Verify(context.Background(), mustPEM(t, h.leaf), mustPEM(t, h.intermediates[0]),
			bytes.NewReader(message), bytes.NewReader(signature), baseTime)
		assertErrorIs(t, err, ErrCertificateExpired)
		ve := assertValidityError(t, err, RoleLeaf, true, baseTime)
		if ve.NotAfter != h.leaf.NotAfter.UTC() {
			t.Errorf("NotAfter mismatch")
		}
	})

	t.Run("leaf not yet valid", func(t *testing.T) {
		h := buildHierarchy(t, 1, timeWindow{notBefore: baseTime.Add(time.Hour), notAfter: baseTime.Add(2 * time.Hour)})
		verifier := mustVerifier(t, defaultPolicy(h))
		message := []byte("future leaf")
		signature := signMessage(t, h.leafKey, message)
		_, err := verifier.Verify(context.Background(), mustPEM(t, h.leaf), mustPEM(t, h.intermediates[0]),
			bytes.NewReader(message), bytes.NewReader(signature), baseTime)
		assertErrorIs(t, err, ErrCertificateNotYetValid)
		assertValidityError(t, err, RoleLeaf, false, baseTime)
	})

	t.Run("intermediate expired first", func(t *testing.T) {
		// Leaf is valid, intermediate already expired.
		h := buildHierarchy(t, 1,
			timeWindow{notBefore: baseTime.Add(-time.Hour), notAfter: baseTime.Add(24 * time.Hour)},
			timeWindow{notBefore: baseTime.Add(-3 * time.Hour), notAfter: baseTime.Add(-time.Minute)})
		verifier := mustVerifier(t, defaultPolicy(h))
		message := []byte("expired intermediate")
		signature := signMessage(t, h.leafKey, message)
		_, err := verifier.Verify(context.Background(), mustPEM(t, h.leaf), mustPEM(t, h.intermediates[0]),
			bytes.NewReader(message), bytes.NewReader(signature), baseTime)
		assertErrorIs(t, err, ErrCertificateExpired)
		assertValidityError(t, err, RoleIntermediate, true, baseTime)
	})

	t.Run("not before boundary equality", func(t *testing.T) {
		h := buildHierarchy(t, 0, timeWindow{notBefore: baseTime, notAfter: baseTime.Add(time.Hour)})
		verifier := mustVerifier(t, defaultPolicy(h))
		message := []byte("boundary not before")
		signature := signMessage(t, h.leafKey, message)
		if _, err := verifier.Verify(context.Background(), mustPEM(t, h.leaf), nil,
			bytes.NewReader(message), bytes.NewReader(signature), baseTime); err != nil {
			t.Fatalf("exact NotBefore must be accepted: %v", err)
		}
	})

	t.Run("not after boundary equality", func(t *testing.T) {
		h := buildHierarchy(t, 0, timeWindow{notBefore: baseTime.Add(-time.Hour), notAfter: baseTime})
		verifier := mustVerifier(t, defaultPolicy(h))
		message := []byte("boundary not after")
		signature := signMessage(t, h.leafKey, message)
		if _, err := verifier.Verify(context.Background(), mustPEM(t, h.leaf), nil,
			bytes.NewReader(message), bytes.NewReader(signature), baseTime); err != nil {
			t.Fatalf("exact NotAfter must be accepted: %v", err)
		}
	})

	t.Run("one nanosecond past not after", func(t *testing.T) {
		h := buildHierarchy(t, 0, timeWindow{notBefore: baseTime.Add(-time.Hour), notAfter: baseTime})
		verifier := mustVerifier(t, defaultPolicy(h))
		message := []byte("just expired")
		signature := signMessage(t, h.leafKey, message)
		_, err := verifier.Verify(context.Background(), mustPEM(t, h.leaf), nil,
			bytes.NewReader(message), bytes.NewReader(signature), baseTime.Add(time.Nanosecond))
		assertErrorIs(t, err, ErrCertificateExpired)
	})
}

func TestCertificateVerifierRootValidity(t *testing.T) {
	tests := []struct {
		name     string
		window   timeWindow
		expired  bool
		sentinel error
	}{
		{
			name:     "expired root",
			window:   timeWindow{notBefore: baseTime.Add(-48 * time.Hour), notAfter: baseTime.Add(-time.Hour)},
			expired:  true,
			sentinel: ErrCertificateExpired,
		},
		{
			name:     "root not yet valid",
			window:   timeWindow{notBefore: baseTime.Add(time.Hour), notAfter: baseTime.Add(48 * time.Hour)},
			expired:  false,
			sentinel: ErrCertificateNotYetValid,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := buildHierarchy(t, 0)
			rootKey := mustGenerateECDSAKey(t)
			rootTemplate := &x509.Certificate{
				SerialNumber:          big.NewInt(7),
				Subject:               pkix.Name{CommonName: "validity-root"},
				NotBefore:             tt.window.notBefore,
				NotAfter:              tt.window.notAfter,
				KeyUsage:              x509.KeyUsageCertSign,
				BasicConstraintsValid: true,
				IsCA:                  true,
			}
			root := mustCreateCert(t, rootTemplate, rootTemplate, &rootKey.PublicKey, rootKey)
			leafTemplate := &x509.Certificate{
				SerialNumber: big.NewInt(8),
				Subject:      pkix.Name{CommonName: "leaf-under-validity-root"},
				NotBefore:    baseTime.Add(-time.Hour),
				NotAfter:     baseTime.Add(24 * time.Hour),
				KeyUsage:     x509.KeyUsageDigitalSignature,
				ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageCodeSigning},
			}
			leaf := mustCreateCert(t, leafTemplate, root, &h.leafKey.PublicKey, rootKey)

			policy := ChainPolicy{
				Roots:     rootPool(root),
				KeyUsages: []x509.ExtKeyUsage{x509.ExtKeyUsageCodeSigning},
			}
			verifier := mustVerifier(t, policy)
			message := []byte("root validity chain")
			signature := signMessage(t, h.leafKey, message)
			_, err := verifier.Verify(context.Background(), mustPEM(t, leaf), nil,
				bytes.NewReader(message), bytes.NewReader(signature), baseTime)
			assertErrorIs(t, err, tt.sentinel)
			ve := assertValidityError(t, err, RoleRoot, tt.expired, baseTime)
			if ve.Subject != root.Subject.String() {
				t.Errorf("expected subject %q, got %q", root.Subject.String(), ve.Subject)
			}
			// The root validity error must not be re-wrapped as an untrusted chain.
			if errors.Is(err, ErrUntrustedChain) {
				t.Errorf("invalid-time root must not be reported as ErrUntrustedChain")
			}
		})
	}
}

func TestCertificateVerifierSigningMode(t *testing.T) {
	h := buildHierarchy(t, 1, timeWindow{
		notBefore: signingInstant.Add(-time.Hour),
		notAfter:  signingInstant.Add(time.Hour),
	})
	authority := newCountingAuthority(func() (time.Time, error) {
		return signingInstant, nil
	})
	policy := defaultPolicy(h)
	policy.Mode = TimeModeSigning
	policy.SigningTimeAuthority = authority.authority
	policy.ClockSkew = time.Minute
	verifier := mustVerifier(t, policy)

	message := []byte("signed in the past")
	signature := signMessage(t, h.leafKey, message)
	// Evaluation time is well after the leaf NotAfter.
	evaluation := baseTime.Add(72 * time.Hour)

	result, err := verifier.Verify(context.Background(), mustPEM(t, h.leaf), mustPEM(t, h.intermediates[0]),
		bytes.NewReader(message), bytes.NewReader(signature), evaluation)
	if err != nil {
		t.Fatalf("signing-mode Verify failed: %v", err)
	}
	if authority.calls.Load() != 1 {
		t.Fatalf("authority must be called exactly once, got %d", authority.calls.Load())
	}
	if result.SigningTime == nil || !result.SigningTime.Equal(signingInstant) {
		t.Fatalf("unexpected signing time in result: %v", result.SigningTime)
	}
	if !result.EffectiveTime.Equal(signingInstant) {
		t.Fatalf("expected effective time %v, got %v", signingInstant, result.EffectiveTime)
	}

	// The authority must receive the SHA-256 of the full message, copies only.
	expectedDigest := sha256.Sum256(message)
	request := authority.lastRequest()
	if request.MessageDigest != expectedDigest {
		t.Fatalf("digest mismatch: got %x, want %x", request.MessageDigest, expectedDigest)
	}
	if !bytes.Equal(request.LeafDER, h.leaf.Raw) {
		t.Fatalf("leaf DER mismatch")
	}
	if !bytes.Equal(request.Signature, signature) {
		t.Fatalf("signature bytes mismatch")
	}
	// Mutating the request slices must not affect anything observable.
	for i := range request.LeafDER {
		request.LeafDER[i] = 0
	}
	for i := range request.Signature {
		request.Signature[i] = 0
	}
	secondMessage := []byte("a different message entirely")
	secondSignature := signMessage(t, h.leafKey, secondMessage)
	_, err = verifier.Verify(context.Background(), mustPEM(t, h.leaf), mustPEM(t, h.intermediates[0]),
		bytes.NewReader(secondMessage), bytes.NewReader(secondSignature), evaluation)
	if err != nil {
		t.Fatalf("second Verify failed: %v", err)
	}
	secondDigest := sha256.Sum256(secondMessage)
	if authority.lastRequest().MessageDigest != secondDigest {
		t.Fatalf("second digest mismatch")
	}
	if secondDigest == expectedDigest {
		t.Fatalf("different messages must produce different digests")
	}
	if authority.calls.Load() != 2 {
		t.Fatalf("authority must not cache across calls, got %d", authority.calls.Load())
	}
}

func TestCertificateVerifierSigningModeFailures(t *testing.T) {
	build := func(t *testing.T, provide func() (time.Time, error), skew time.Duration) (*CertificateVerifier, *certHierarchy, *countingAuthority) {
		// A wide leaf window so that the clock-skew boundary tests reach (or
		// stop before) the time check they are interested in.
		h := buildHierarchy(t, 1, timeWindow{
			notBefore: signingInstant.Add(-2 * time.Hour),
			notAfter:  baseTime.Add(2 * time.Hour),
		})
		authority := newCountingAuthority(provide)
		policy := defaultPolicy(h)
		policy.Mode = TimeModeSigning
		policy.SigningTimeAuthority = authority.authority
		policy.ClockSkew = skew
		return mustVerifier(t, policy), h, authority
	}

	t.Run("authority error", func(t *testing.T) {
		verifier, h, authority := build(t, func() (time.Time, error) {
			return time.Time{}, errors.New("timestamp service down")
		}, 0)
		message := []byte("authority fails")
		signature := signMessage(t, h.leafKey, message)
		_, err := verifier.Verify(context.Background(), mustPEM(t, h.leaf), mustPEM(t, h.intermediates[0]),
			bytes.NewReader(message), bytes.NewReader(signature), baseTime)
		assertErrorIs(t, err, ErrSigningTimeUnavailable)
		if !strings.Contains(err.Error(), "timestamp service down") {
			t.Errorf("authority error chain must be retained, got %v", err)
		}
		if authority.calls.Load() != 1 {
			t.Errorf("authority called %d times", authority.calls.Load())
		}
	})

	t.Run("authority zero time", func(t *testing.T) {
		verifier, h, _ := build(t, func() (time.Time, error) { return time.Time{}, nil }, 0)
		message := []byte("zero time")
		signature := signMessage(t, h.leafKey, message)
		_, err := verifier.Verify(context.Background(), mustPEM(t, h.leaf), mustPEM(t, h.intermediates[0]),
			bytes.NewReader(message), bytes.NewReader(signature), baseTime)
		assertErrorIs(t, err, ErrSigningTimeUnavailable)
	})

	t.Run("signing time in future", func(t *testing.T) {
		future := baseTime.Add(time.Hour)
		verifier, h, _ := build(t, func() (time.Time, error) { return future, nil }, time.Minute)
		message := []byte("future signature")
		signature := signMessage(t, h.leafKey, message)
		_, err := verifier.Verify(context.Background(), mustPEM(t, h.leaf), mustPEM(t, h.intermediates[0]),
			bytes.NewReader(message), bytes.NewReader(signature), baseTime)
		assertErrorIs(t, err, ErrSigningTimeInFuture)
	})

	t.Run("clock skew boundary accepted and one nanosecond rejected", func(t *testing.T) {
		skew := time.Minute
		verifier, h, _ := build(t, func() (time.Time, error) { return baseTime.Add(skew), nil }, skew)
		message := []byte("at the skew boundary")
		signature := signMessage(t, h.leafKey, message)
		if _, err := verifier.Verify(context.Background(), mustPEM(t, h.leaf), mustPEM(t, h.intermediates[0]),
			bytes.NewReader(message), bytes.NewReader(signature), baseTime); err != nil {
			t.Fatalf("signing time exactly at evaluation+skew must pass: %v", err)
		}

		verifier2, h2, _ := build(t, func() (time.Time, error) { return baseTime.Add(skew + time.Nanosecond), nil }, skew)
		signature2 := signMessage(t, h2.leafKey, message)
		_, err := verifier2.Verify(context.Background(), mustPEM(t, h2.leaf), mustPEM(t, h2.intermediates[0]),
			bytes.NewReader(message), bytes.NewReader(signature2), baseTime)
		assertErrorIs(t, err, ErrSigningTimeInFuture)
	})

	t.Run("no fallback to current time after authority error", func(t *testing.T) {
		verifier, h, _ := build(t, func() (time.Time, error) {
			return time.Time{}, errors.New("boom")
		}, 0)
		message := []byte("no fallback")
		signature := signMessage(t, h.leafKey, message)
		result, err := verifier.Verify(context.Background(), mustPEM(t, h.leaf), mustPEM(t, h.intermediates[0]),
			bytes.NewReader(message), bytes.NewReader(signature), baseTime)
		if result != nil {
			t.Errorf("result must be nil on failure")
		}
		assertErrorIs(t, err, ErrSigningTimeUnavailable)
	})
}

func TestCertificateVerifierInputErrors(t *testing.T) {
	h := buildHierarchy(t, 1)
	verifier := mustVerifier(t, defaultPolicy(h))
	message := []byte("input checks")
	signature := signMessage(t, h.leafKey, message)
	leafPEM := mustPEM(t, h.leaf)
	intermediatePEM := mustPEM(t, h.intermediates[0])

	t.Run("nil context", func(t *testing.T) {
		//nolint:staticcheck // intentionally passing nil context
		_, err := verifier.Verify(nil, leafPEM, intermediatePEM, bytes.NewReader(message), bytes.NewReader(signature), baseTime)
		assertErrorIs(t, err, ErrInvalidInput)
	})
	t.Run("zero evaluation time", func(t *testing.T) {
		_, err := verifier.Verify(context.Background(), leafPEM, intermediatePEM, bytes.NewReader(message), bytes.NewReader(signature), time.Time{})
		assertErrorIs(t, err, ErrInvalidInput)
	})
	t.Run("nil message", func(t *testing.T) {
		_, err := verifier.Verify(context.Background(), leafPEM, intermediatePEM, nil, bytes.NewReader(signature), baseTime)
		assertErrorIs(t, err, ErrInvalidInput)
	})
	t.Run("nil signature", func(t *testing.T) {
		_, err := verifier.Verify(context.Background(), leafPEM, intermediatePEM, bytes.NewReader(message), nil, baseTime)
		assertErrorIs(t, err, ErrInvalidInput)
	})
	t.Run("empty leaf PEM", func(t *testing.T) {
		_, err := verifier.Verify(context.Background(), nil, intermediatePEM, bytes.NewReader(message), bytes.NewReader(signature), baseTime)
		assertErrorIs(t, err, ErrInvalidInput)
	})
	t.Run("two leaf certificates", func(t *testing.T) {
		twoLeaf := mustConcatPEM(t, h.leaf, h.leaf)
		_, err := verifier.Verify(context.Background(), twoLeaf, intermediatePEM, bytes.NewReader(message), bytes.NewReader(signature), baseTime)
		assertErrorIs(t, err, ErrInvalidInput)
	})
}

func TestCertificateVerifierMalformedCertificates(t *testing.T) {
	h := buildHierarchy(t, 1)
	verifier := mustVerifier(t, defaultPolicy(h))
	message := []byte("malformed")
	signature := signMessage(t, h.leafKey, message)
	intermediatePEM := mustPEM(t, h.intermediates[0])

	t.Run("trusted leaf DER", func(t *testing.T) {
		truncated := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: h.leaf.Raw[:len(h.leaf.Raw)/2]})
		_, err := verifier.Verify(context.Background(), truncated, intermediatePEM, bytes.NewReader(message), bytes.NewReader(signature), baseTime)
		assertErrorIs(t, err, ErrMalformedCertificate)
		// The x509 parse failure must remain reachable through the error chain.
		if !strings.Contains(err.Error(), "x509") && !strings.Contains(err.Error(), "asn1") {
			t.Errorf("expected retained parse error text, got %v", err)
		}
	})
	t.Run("non certificate block", func(t *testing.T) {
		block := pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: []byte{1, 2, 3}})
		_, err := verifier.Verify(context.Background(), block, intermediatePEM, bytes.NewReader(message), bytes.NewReader(signature), baseTime)
		assertErrorIs(t, err, ErrMalformedCertificate)
	})
	t.Run("trailing garbage", func(t *testing.T) {
		garbage := append(append([]byte{}, mustPEM(t, h.leaf)...), []byte("not pem at all")...)
		_, err := verifier.Verify(context.Background(), garbage, intermediatePEM, bytes.NewReader(message), bytes.NewReader(signature), baseTime)
		assertErrorIs(t, err, ErrMalformedCertificate)
	})
	t.Run("empty certificate block", func(t *testing.T) {
		empty := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: nil})
		_, err := verifier.Verify(context.Background(), empty, intermediatePEM, bytes.NewReader(message), bytes.NewReader(signature), baseTime)
		assertErrorIs(t, err, ErrMalformedCertificate)
	})
	t.Run("malformed intermediate retains parse error", func(t *testing.T) {
		badIntermediate := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: []byte{0x30, 0x00}})
		_, err := verifier.Verify(context.Background(), mustPEM(t, h.leaf), badIntermediate, bytes.NewReader(message), bytes.NewReader(signature), baseTime)
		assertErrorIs(t, err, ErrMalformedCertificate)
		// The concrete x509.ParseCertificate failure must be retained by %w and
		// must not be reclassified as a chain-structure error.
		if errors.Is(err, ErrInvalidChain) {
			t.Fatalf("parse failure must not be reported as ErrInvalidChain: %v", err)
		}
		if !strings.Contains(err.Error(), "x509") && !strings.Contains(err.Error(), "asn1") {
			t.Fatalf("expected retained parse error text, got %v", err)
		}
	})
	t.Run("parse failure precedes chain structure", func(t *testing.T) {
		// A malformed block must surface as a parse error even though the same
		// slot would also be a structural problem if it parsed.
		bad := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: h.intermediates[0].Raw[:4]})
		_, err := verifier.Verify(context.Background(), mustPEM(t, h.leaf), bad, bytes.NewReader(message), bytes.NewReader(signature), baseTime)
		assertErrorIs(t, err, ErrMalformedCertificate)
	})
}

func TestCertificateVerifierChainStructure(t *testing.T) {
	h := buildHierarchy(t, 1)
	verifier := mustVerifier(t, defaultPolicy(h))
	message := []byte("chain structure")
	signature := signMessage(t, h.leafKey, message)
	leafPEM := mustPEM(t, h.leaf)

	t.Run("root among intermediates", func(t *testing.T) {
		_, err := verifier.Verify(context.Background(), leafPEM, mustPEM(t, h.root),
			bytes.NewReader(message), bytes.NewReader(signature), baseTime)
		assertErrorIs(t, err, ErrInvalidChain)
	})
	t.Run("duplicate intermediate", func(t *testing.T) {
		dupes := mustConcatPEM(t, h.intermediates[0], h.intermediates[0])
		_, err := verifier.Verify(context.Background(), leafPEM, dupes,
			bytes.NewReader(message), bytes.NewReader(signature), baseTime)
		assertErrorIs(t, err, ErrInvalidChain)
	})
	t.Run("out of order intermediates", func(t *testing.T) {
		h2 := buildHierarchy(t, 2)
		verifier2 := mustVerifier(t, defaultPolicy(h2))
		signature2 := signMessage(t, h2.leafKey, message)
		// Reversing [I0, I1] makes I1 the apparent leaf issuer, which it is
		// not, so the chain structure must be rejected.
		reversed := mustConcatPEM(t, h2.intermediates[1], h2.intermediates[0])
		_, err := verifier2.Verify(context.Background(), mustPEM(t, h2.leaf), reversed,
			bytes.NewReader(message), bytes.NewReader(signature2), baseTime)
		assertErrorIs(t, err, ErrInvalidChain)
	})
	t.Run("unrelated intermediate", func(t *testing.T) {
		other := buildHierarchy(t, 1)
		_, err := verifier.Verify(context.Background(), leafPEM, mustPEM(t, other.intermediates[0]),
			bytes.NewReader(message), bytes.NewReader(signature), baseTime)
		assertErrorIs(t, err, ErrInvalidChain)
	})
	t.Run("leaf is CA", func(t *testing.T) {
		caLeaf := buildLeafAsCA(t, h)
		_, err := verifier.Verify(context.Background(), mustPEM(t, caLeaf), mustPEM(t, h.intermediates[0]),
			bytes.NewReader(message), bytes.NewReader(signature), baseTime)
		assertErrorIs(t, err, ErrLeafIsCA)
	})
}

func TestCertificateVerifierTooManyIntermediates(t *testing.T) {
	h := buildHierarchy(t, 3)
	policy := defaultPolicy(h)
	policy.MaxIntermediates = 2
	verifier := mustVerifier(t, policy)
	message := []byte("long chain")
	signature := signMessage(t, h.leafKey, message)
	_, err := verifier.Verify(context.Background(), mustPEM(t, h.leaf), mustConcatPEM(t, h.intermediates...),
		bytes.NewReader(message), bytes.NewReader(signature), baseTime)
	assertErrorIs(t, err, ErrTooManyIntermediates)

	// Exactly at the limit succeeds structurally and reaches trust validation.
	policy.MaxIntermediates = 3
	verifier = mustVerifier(t, policy)
	if _, err := verifier.Verify(context.Background(), mustPEM(t, h.leaf), mustConcatPEM(t, h.intermediates...),
		bytes.NewReader(message), bytes.NewReader(signature), baseTime); err != nil {
		t.Fatalf("chain at the limit must be accepted: %v", err)
	}

	// The hard ceiling of 16 cannot be exceeded even when the policy asks for it.
	_, err = NewCertificateVerifier(ChainPolicy{
		Roots:            rootPool(h.root),
		KeyUsages:        []x509.ExtKeyUsage{x509.ExtKeyUsageCodeSigning},
		MaxIntermediates: 17,
	})
	assertErrorIs(t, err, ErrInvalidPolicy)
}

func TestCertificateVerifierSignatureFailures(t *testing.T) {
	h := buildHierarchy(t, 1)
	verifier := mustVerifier(t, defaultPolicy(h))
	intermediatePEM := mustPEM(t, h.intermediates[0])
	message := []byte("signature failures")
	goodSignature := signMessage(t, h.leafKey, message)

	t.Run("corrupt signature", func(t *testing.T) {
		bad := append([]byte{}, goodSignature...)
		bad[len(bad)-1] ^= 0xFF
		_, err := verifier.Verify(context.Background(), mustPEM(t, h.leaf), intermediatePEM,
			bytes.NewReader(message), bytes.NewReader(bad), baseTime)
		assertErrorIs(t, err, ErrSignatureInvalid)
	})
	t.Run("empty signature", func(t *testing.T) {
		_, err := verifier.Verify(context.Background(), mustPEM(t, h.leaf), intermediatePEM,
			bytes.NewReader(message), bytes.NewReader(nil), baseTime)
		assertErrorIs(t, err, ErrSignatureInvalid)
	})
	t.Run("empty message with non empty signature", func(t *testing.T) {
		emptySignature := signMessage(t, h.leafKey, nil)
		result, err := verifier.Verify(context.Background(), mustPEM(t, h.leaf), intermediatePEM,
			bytes.NewReader(nil), bytes.NewReader(emptySignature), baseTime)
		if err != nil {
			t.Fatalf("empty message must verify: %v", err)
		}
		emptyDigest := sha256.Sum256(nil)
		_ = result
		// Re-derive independently to confirm the empty-string digest path.
		if fmt.Sprintf("%x", emptyDigest) != fmt.Sprintf("%x", sha256.Sum256([]byte{})) {
			t.Fatal("nil and empty digests differ")
		}
	})
	t.Run("signature too large", func(t *testing.T) {
		oversized := make([]byte, maxSignatureBytes+1)
		_, err := verifier.Verify(context.Background(), mustPEM(t, h.leaf), intermediatePEM,
			bytes.NewReader(message), bytes.NewReader(oversized), baseTime)
		assertErrorIs(t, err, ErrSignatureTooLarge)
	})
	t.Run("signature at exactly the limit is not rejected for size", func(t *testing.T) {
		// Random bytes at the size limit fail signature math, not the size check.
		limitSized := make([]byte, maxSignatureBytes)
		_, err := verifier.Verify(context.Background(), mustPEM(t, h.leaf), intermediatePEM,
			bytes.NewReader(message), bytes.NewReader(limitSized), baseTime)
		if errors.Is(err, ErrSignatureTooLarge) {
			t.Fatalf("1 MiB signature must not hit the size sentinel")
		}
		assertErrorIs(t, err, ErrSignatureInvalid)
	})
	t.Run("reader counts all message bytes", func(t *testing.T) {
		counter := &countingReader{r: bytes.NewReader(message)}
		if _, err := verifier.Verify(context.Background(), mustPEM(t, h.leaf), intermediatePEM,
			counter, bytes.NewReader(goodSignature), baseTime); err != nil {
			t.Fatalf("Verify failed: %v", err)
		}
		if counter.count != int64(len(message)) {
			t.Fatalf("expected %d bytes read, got %d", len(message), counter.count)
		}
	})
	t.Run("unsupported leaf key type", func(t *testing.T) {
		// A real parsed certificate can only carry key types the stdlib x509
		// package supports, all of which LoadVerifierWithOpts also handles.
		// To exercise the unsupported-key branch deterministically, call the
		// in-package signature stage with a leaf whose PublicKey is a type the
		// loader does not recognize.
		leaf := *h.leaf
		leaf.PublicKey = &unsupportedPublicKey{}
		_, _, err := verifier.verifySignature(context.Background(), &leaf,
			bytes.NewReader(message), bytes.NewReader(goodSignature))
		assertErrorIs(t, err, ErrUnsupportedKey)
	})
}

// unsupportedPublicKey is a crypto.PublicKey that no verifier in the package
// knows how to load. It exists only to drive the ErrUnsupportedKey branch.
type unsupportedPublicKey struct{}

func (*unsupportedPublicKey) Equal(crypto.PublicKey) bool { return false }

func TestCertificateVerifierUntrusted(t *testing.T) {
	h := buildHierarchy(t, 1)
	message := []byte("trust failures")
	signature := signMessage(t, h.leafKey, message)

	t.Run("untrusted root", func(t *testing.T) {
		other := buildHierarchy(t, 0)
		policy := ChainPolicy{
			Roots:     rootPool(other.root),
			KeyUsages: []x509.ExtKeyUsage{x509.ExtKeyUsageCodeSigning},
		}
		verifier := mustVerifier(t, policy)
		_, err := verifier.Verify(context.Background(), mustPEM(t, h.leaf), mustPEM(t, h.intermediates[0]),
			bytes.NewReader(message), bytes.NewReader(signature), baseTime)
		assertErrorIs(t, err, ErrUntrustedChain)
		var unknown x509.UnknownAuthorityError
		if !errors.As(err, &unknown) {
			t.Fatalf("original x509 error must remain reachable, got %v", err)
		}
	})

	t.Run("eku mismatch", func(t *testing.T) {
		serverLeaf := buildLeafWithEKU(t, h, []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth})
		verifier := mustVerifier(t, defaultPolicy(h))
		signature2 := signMessage(t, h.leafKey, message)
		_, err := verifier.Verify(context.Background(), mustPEM(t, serverLeaf), mustPEM(t, h.intermediates[0]),
			bytes.NewReader(message), bytes.NewReader(signature2), baseTime)
		assertErrorIs(t, err, ErrUntrustedChain)
		var invalid x509.CertificateInvalidError
		if !errors.As(err, &invalid) {
			t.Fatalf("x509.CertificateInvalidError must remain reachable, got %v", err)
		}
	})
}

func TestCertificateVerifierPriority(t *testing.T) {
	t.Run("signature error before expiry", func(t *testing.T) {
		h := buildHierarchy(t, 1, timeWindow{notBefore: baseTime.Add(-2 * time.Hour), notAfter: baseTime.Add(-time.Hour)})
		verifier := mustVerifier(t, defaultPolicy(h))
		message := []byte("both broken")
		goodSignature := signMessage(t, h.leafKey, message)
		bad := append([]byte{}, goodSignature...)
		bad[0] ^= 0xFF
		_, err := verifier.Verify(context.Background(), mustPEM(t, h.leaf), mustPEM(t, h.intermediates[0]),
			bytes.NewReader(message), bytes.NewReader(bad), baseTime)
		assertErrorIs(t, err, ErrSignatureInvalid)
	})

	t.Run("chain error before signature error", func(t *testing.T) {
		h := buildHierarchy(t, 1)
		verifier := mustVerifier(t, defaultPolicy(h))
		message := []byte("both broken two")
		goodSignature := signMessage(t, h.leafKey, message)
		bad := append([]byte{}, goodSignature...)
		bad[0] ^= 0xFF
		// Put the root into the intermediate slot: chain failure.
		_, err := verifier.Verify(context.Background(), mustPEM(t, h.leaf), mustPEM(t, h.root),
			bytes.NewReader(message), bytes.NewReader(bad), baseTime)
		assertErrorIs(t, err, ErrInvalidChain)
	})

	t.Run("deterministic error category across repeats", func(t *testing.T) {
		h := buildHierarchy(t, 1)
		verifier := mustVerifier(t, defaultPolicy(h))
		message := []byte("repeatable")
		signature := signMessage(t, h.leafKey, message)
		for i := 0; i < 5; i++ {
			_, err := verifier.Verify(context.Background(), mustPEM(t, h.leaf), mustPEM(t, h.root),
				bytes.NewReader(message), bytes.NewReader(signature), baseTime)
			assertErrorIs(t, err, ErrInvalidChain)
		}
	})
}

func TestCertificateVerifierImmutability(t *testing.T) {
	h := buildHierarchy(t, 1)
	policy := defaultPolicy(h)
	keyUsages := []x509.ExtKeyUsage{x509.ExtKeyUsageCodeSigning}
	policy.KeyUsages = keyUsages
	verifier := mustVerifier(t, policy)
	message := []byte("immutable")
	signature := signMessage(t, h.leafKey, message)

	// Mutate the caller's KeyUsages backing array.
	keyUsages[0] = x509.ExtKeyUsageServerAuth
	// Add an unrelated certificate to the caller's root pool.
	policy.Roots.AddCert(buildHierarchy(t, 0).root)
	for i := 0; i < 3; i++ {
		result, err := verifier.Verify(context.Background(), mustPEM(t, h.leaf), mustPEM(t, h.intermediates[0]),
			bytes.NewReader(message), bytes.NewReader(signature), baseTime)
		if err != nil {
			t.Fatalf("verifier behavior changed after input mutation: %v", err)
		}
		if len(result.Chain) != 3 {
			t.Fatalf("unexpected chain after mutation: %d", len(result.Chain))
		}
	}

	// PEM inputs must not be modified by Verify.
	leafPEM := mustPEM(t, h.leaf)
	intermediatePEM := mustPEM(t, h.intermediates[0])
	leafCopy := append([]byte{}, leafPEM...)
	intermediateCopy := append([]byte{}, intermediatePEM...)
	_, _ = verifier.Verify(context.Background(), leafPEM, intermediatePEM,
		bytes.NewReader(message), bytes.NewReader(signature), baseTime)
	if !bytes.Equal(leafPEM, leafCopy) || !bytes.Equal(intermediatePEM, intermediateCopy) {
		t.Fatal("Verify modified its PEM inputs")
	}
}

func TestCertificateVerifierConcurrency(t *testing.T) {
	h := buildHierarchy(t, 1)
	verifier := mustVerifier(t, defaultPolicy(h))
	message := []byte("concurrent verification")
	signature := signMessage(t, h.leafKey, message)
	leafPEM := mustPEM(t, h.leaf)
	intermediatePEM := mustPEM(t, h.intermediates[0])

	const goroutines = 64
	const iterations = 100
	var wg sync.WaitGroup
	errCh := make(chan error, goroutines)
	wg.Add(goroutines)
	for g := 0; g < goroutines; g++ {
		go func() {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				result, err := verifier.Verify(context.Background(), leafPEM, intermediatePEM,
					bytes.NewReader(message), bytes.NewReader(signature), baseTime)
				if err != nil {
					errCh <- err
					return
				}
				if len(result.Chain) != 3 {
					errCh <- fmt.Errorf("unexpected chain length %d", len(result.Chain))
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Fatalf("concurrent Verify failed: %v", err)
	}
}

func TestCertificateVerifierContextCanceled(t *testing.T) {
	h := buildHierarchy(t, 1)
	authority := newCountingAuthority(nil)
	authority.block = true
	policy := defaultPolicy(h)
	policy.Mode = TimeModeSigning
	policy.SigningTimeAuthority = authority.authority
	verifier := mustVerifier(t, policy)

	message := []byte("cancel me")
	signature := signMessage(t, h.leafKey, message)
	ctx, cancel := context.WithCancel(context.Background())

	errCh := make(chan error, 1)
	go func() {
		_, err := verifier.Verify(ctx, mustPEM(t, h.leaf), mustPEM(t, h.intermediates[0]),
			bytes.NewReader(message), bytes.NewReader(signature), baseTime)
		errCh <- err
	}()
	<-authority.entered
	cancel()
	err := <-errCh
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}

	// A deadline exceeding at an input stage must surface directly as well.
	expiredCtx, expiredCancel := context.WithDeadline(context.Background(), baseTime.Add(-time.Hour))
	defer expiredCancel()
	_, err = verifier.Verify(expiredCtx, mustPEM(t, h.leaf), mustPEM(t, h.intermediates[0]),
		bytes.NewReader(message), bytes.NewReader(signature), baseTime)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected context.DeadlineExceeded, got %v", err)
	}

	// Cancellation observed after the authority returns successfully must win
	// over the signing time and never fall back to evaluation-time semantics.
	canceledCtx, cancel2 := context.WithCancel(context.Background())
	successAuthority := newCountingAuthority(func() (time.Time, error) {
		cancel2()
		return signingInstant, nil
	})
	policy2 := defaultPolicy(h)
	policy2.Mode = TimeModeSigning
	policy2.SigningTimeAuthority = successAuthority.authority
	verifier2 := mustVerifier(t, policy2)
	result, err := verifier2.Verify(canceledCtx, mustPEM(t, h.leaf), mustPEM(t, h.intermediates[0]),
		bytes.NewReader(message), bytes.NewReader(signature), baseTime)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled after authority returned, got %v", err)
	}
	if result != nil {
		t.Fatalf("no partial result on cancellation")
	}
}

func TestCertificateVerifierResultIsNilOnFailure(t *testing.T) {
	h := buildHierarchy(t, 1)
	verifier := mustVerifier(t, defaultPolicy(h))
	result, err := verifier.Verify(context.Background(), mustPEM(t, h.leaf), mustPEM(t, h.intermediates[0]),
		bytes.NewReader([]byte("message")), bytes.NewReader([]byte("not a signature")), baseTime)
	if err == nil {
		t.Fatal("expected verification failure")
	}
	if result != nil {
		t.Fatal("result must be nil on failure")
	}
	assertErrorIs(t, err, ErrSignatureInvalid)
}

func FuzzCertificateVerifier(f *testing.F) {
	h := buildHierarchy(f, 1)
	verifier, err := NewCertificateVerifier(ChainPolicy{
		Roots:     rootPool(h.root),
		KeyUsages: []x509.ExtKeyUsage{x509.ExtKeyUsageCodeSigning},
	})
	if err != nil {
		f.Fatal(err)
	}
	message := []byte("fuzz message")
	signature := signMessage(f, h.leafKey, message)
	validLeaf := mustPEM(f, h.leaf)
	validIntermediate := mustPEM(f, h.intermediates[0])

	f.Add([]byte(""), []byte(""), []byte(""), int64(0))
	f.Add(validLeaf, validIntermediate, signature, baseTime.UnixNano())
	f.Add([]byte("-----BEGIN CERTIFICATE-----\n-----END CERTIFICATE-----\n"), []byte(""), []byte(""), baseTime.UnixNano())
	f.Add(validLeaf[:len(validLeaf)/2], validIntermediate, signature, baseTime.UnixNano())

	f.Fuzz(func(t *testing.T, leafPEM, intermediatesPEM, signatureBytes []byte, evalNanos int64) {
		evaluation := time.Unix(0, evalNanos).UTC()
		// The verifier must never panic, regardless of the garbage inputs.
		// A nil evaluation time is itself an expected, handled error.
		_, _ = verifier.Verify(
			context.Background(),
			leafPEM,
			intermediatesPEM,
			bytes.NewReader(message),
			bytes.NewReader(signatureBytes),
			evaluation,
		)
	})
}
