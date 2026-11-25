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
	"crypto/sha256"
	_ "crypto/sha512" // register SHA-384 and SHA-512 so crypto.Hash.Available reports them as usable
	"crypto/x509"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"time"

	"github.com/sigstore/sigstore/pkg/cryptoutils"
	"github.com/sigstore/sigstore/pkg/signature/options"
)

const (
	// maxIntermediateCerts is the hard upper bound on the number of
	// intermediate certificates accepted in a chain.
	maxIntermediateCerts = 16
	// defaultMaxIntermediateCerts is the ChainPolicy.MaxIntermediates value
	// used when the field is left at its zero value.
	defaultMaxIntermediateCerts = 8
	// maxClockSkew is the hard upper bound on ChainPolicy.ClockSkew.
	maxClockSkew = 5 * time.Minute
	// maxSignatureBytes is the hard upper bound on the number of signature
	// bytes that Verify is willing to read.
	maxSignatureBytes = 1 << 20 // 1 MiB
)

// TimeMode selects which instant is used when evaluating certificate validity
// and building the trusted path.
type TimeMode int

const (
	// TimeModeCurrent evaluates certificate validity at the caller-supplied
	// evaluation time. No SigningTimeAuthority is consulted.
	TimeModeCurrent TimeMode = iota
	// TimeModeSigning derives the effective time from the
	// SigningTimeAuthority instead of using the evaluation time. The
	// evaluation time is only used to reject signing times that lie in the
	// future (subject to the configured clock skew).
	TimeModeSigning
)

// CertRole labels the position a certificate occupies in a verified chain.
type CertRole string

const (
	// RoleLeaf is the end-entity certificate carrying the verified public key.
	RoleLeaf CertRole = "leaf"
	// RoleIntermediate is a CA certificate between the leaf and the root.
	RoleIntermediate CertRole = "intermediate"
	// RoleRoot is the self-signed trust anchor.
	RoleRoot CertRole = "root"
)

// Sentinel errors returned by the certificate verifier. All of them can be
// matched with errors.Is.
var (
	// ErrInvalidPolicy indicates that a ChainPolicy passed to
	// NewCertificateVerifier is invalid.
	ErrInvalidPolicy = errors.New("invalid certificate verifier policy")
	// ErrInvalidInput indicates that the arguments passed to Verify are
	// invalid independently of their contents (nil readers, zero evaluation
	// time, a leaf PEM that does not contain exactly one certificate, ...).
	ErrInvalidInput = errors.New("invalid certificate verifier input")
	// ErrMalformedCertificate indicates that a PEM block cannot be decoded
	// into an x509 certificate. The underlying parse error is retained.
	ErrMalformedCertificate = errors.New("malformed certificate")
	// ErrInvalidChain indicates that the supplied certificates do not form a
	// structurally valid chain: bad ordering, broken issuer relations,
	// duplicate certificates, non-CA intermediates or a root in the
	// intermediate list.
	ErrInvalidChain = errors.New("invalid certificate chain")
	// ErrLeafIsCA indicates that the leaf certificate has IsCA set.
	ErrLeafIsCA = errors.New("leaf certificate must not be a CA")
	// ErrTooManyIntermediates indicates that more than the configured
	// MaxIntermediates certificates were supplied as intermediates.
	ErrTooManyIntermediates = errors.New("too many intermediate certificates")
	// ErrSignatureTooLarge indicates that the signature exceeds the 1 MiB
	// read limit.
	ErrSignatureTooLarge = errors.New("signature exceeds 1 MiB limit")
	// ErrSignatureInvalid indicates that the signature bytes do not verify
	// against the message, or that the signature could not be read.
	ErrSignatureInvalid = errors.New("invalid signature")
	// ErrUnsupportedKey indicates that the leaf public key algorithm is not
	// supported by the signature package.
	ErrUnsupportedKey = errors.New("unsupported leaf public key type")
	// ErrSigningTimeUnavailable indicates that the SigningTimeAuthority
	// returned an error or a zero time.
	ErrSigningTimeUnavailable = errors.New("signing time unavailable")
	// ErrSigningTimeInFuture indicates that the reported signing time is more
	// than ClockSkew later than the evaluation time.
	ErrSigningTimeInFuture = errors.New("signing time is in the future")
	// ErrCertificateExpired is the sentinel unwrapped by a
	// CertificateValidityError when a certificate is past its NotAfter.
	ErrCertificateExpired = errors.New("certificate has expired")
	// ErrCertificateNotYetValid is the sentinel unwrapped by a
	// CertificateValidityError when a certificate is before its NotBefore.
	ErrCertificateNotYetValid = errors.New("certificate is not yet valid")
	// ErrUntrustedChain indicates that the chain could not be anchored to a
	// trusted root, including EKU or key-usage mismatches. The underlying
	// x509 verification error is retained.
	ErrUntrustedChain = errors.New("untrusted certificate chain")
)

// SigningTimeRequest carries the evidence handed to a SigningTimeAuthority.
// The byte slices are copies: a callback is free to modify them without
// affecting the surrounding verification.
type SigningTimeRequest struct {
	// LeafDER is the DER encoding of the leaf certificate.
	LeafDER []byte
	// MessageDigest is the SHA-256 digest of every byte of the raw message,
	// independent of ChainPolicy.Hash. Tail bytes not consumed by signature
	// verification are included.
	MessageDigest [32]byte
	// Signature is the raw signature byte string.
	Signature []byte
}

// SigningTimeAuthority resolves the instant at which a signature was produced.
//
// Implementations must be safe for concurrent use when a single
// CertificateVerifier is shared across goroutines: Verify may invoke the
// authority concurrently and never serializes calls to it. Implementations
// must not perform certificate revocation checks; revocation decisions are the
// caller's responsibility.
type SigningTimeAuthority func(ctx context.Context, request SigningTimeRequest) (time.Time, error)

// ChainPolicy configures a CertificateVerifier. A CertificateVerifier copies
// the mutable parts of the policy at construction, so later mutations of the
// caller's Roots pool or KeyUsages slice do not affect the verifier.
type ChainPolicy struct {
	// Roots is the set of trusted root certificates. Required.
	Roots *x509.CertPool
	// KeyUsages is the set of extended key usages accepted on the leaf.
	// Required and must contain at least one entry.
	KeyUsages []x509.ExtKeyUsage
	// Hash selects the digest algorithm used to verify the signature. A zero
	// value means SHA-256. Only SHA-256, SHA-384 and SHA-512 are allowed.
	Hash crypto.Hash
	// MaxIntermediates bounds the number of accepted intermediate
	// certificates. A zero value means 8. Values must be between 1 and 16.
	MaxIntermediates int
	// ClockSkew is extra tolerance applied only when comparing the reported
	// signing time against the evaluation time. It never widens certificate
	// validity windows. A zero value disables extra tolerance. Values must
	// not exceed 5 minutes.
	ClockSkew time.Duration
	// Mode selects evaluation-time or signing-time semantics.
	Mode TimeMode
	// SigningTimeAuthority resolves the signing time in TimeModeSigning. It
	// must be non-nil in that mode and is never called in TimeModeCurrent.
	SigningTimeAuthority SigningTimeAuthority
}

// CertificateVerifier verifies X.509 certificate chains together with the
// signature produced by the leaf key. A verifier is immutable after
// construction and safe for concurrent use.
//
// The verifier never consults CRLs, OCSP responders, the network, environment
// variables or configuration files. Certificate revocation handling is left
// entirely to the caller.
type CertificateVerifier struct {
	roots             *x509.CertPool
	keyUsages         []x509.ExtKeyUsage
	hash              crypto.Hash
	maxIntermediates  int
	clockSkew         time.Duration
	mode              TimeMode
	signingTimeSource SigningTimeAuthority
}

// CertificateValidityError reports that a certificate in the chain is outside
// its validity window at the effective time. Unwrap returns
// ErrCertificateExpired or ErrCertificateNotYetValid depending on Expired.
type CertificateValidityError struct {
	// Role identifies the offending certificate.
	Role CertRole
	// Subject is the string representation of the certificate subject.
	Subject string
	// NotBefore is the start of the validity window, in UTC.
	NotBefore time.Time
	// NotAfter is the end of the validity window, in UTC.
	NotAfter time.Time
	// EffectiveTime is the instant the chain was evaluated at, in UTC.
	EffectiveTime time.Time
	// Expired is true when EffectiveTime is after NotAfter and false when it
	// is before NotBefore.
	Expired bool
}

// Error renders the validity failure with UTC RFC3339 timestamps.
func (e *CertificateValidityError) Error() string {
	if e.Expired {
		return fmt.Sprintf("%s certificate %q expired: notAfter %s is before effective time %s",
			e.Role, e.Subject, e.NotAfter.Format(time.RFC3339), e.EffectiveTime.Format(time.RFC3339))
	}
	return fmt.Sprintf("%s certificate %q not yet valid: notBefore %s is after effective time %s",
		e.Role, e.Subject, e.NotBefore.Format(time.RFC3339), e.EffectiveTime.Format(time.RFC3339))
}

// Unwrap returns ErrCertificateExpired when Expired is set and
// ErrCertificateNotYetValid otherwise.
func (e *CertificateValidityError) Unwrap() error {
	if e.Expired {
		return ErrCertificateExpired
	}
	return ErrCertificateNotYetValid
}

// VerificationResult is returned by Verify on success.
type VerificationResult struct {
	// Leaf is the verified end-entity certificate.
	Leaf *x509.Certificate
	// Chain holds the certificates from the leaf up to the trust anchor.
	Chain []*x509.Certificate
	// Roles holds one CertRole per Chain entry, in the same order.
	Roles []CertRole
	// Mode is the time mode that was applied.
	Mode TimeMode
	// EffectiveTime is the instant used for certificate validation, in UTC.
	EffectiveTime time.Time
	// SigningTime carries the authority-reported signing time in
	// TimeModeSigning. It is nil in TimeModeCurrent.
	SigningTime *time.Time
}

// NewCertificateVerifier validates policy and returns an immutable
// CertificateVerifier. The Roots pool is cloned and the KeyUsages slice is
// copied, so subsequent caller mutations cannot change verifier behavior. Any
// policy violation is wrapped around ErrInvalidPolicy.
func NewCertificateVerifier(policy ChainPolicy) (*CertificateVerifier, error) {
	if policy.Roots == nil || len(policy.Roots.Subjects()) == 0 {
		return nil, fmt.Errorf("%w: Roots must be a non-empty certificate pool", ErrInvalidPolicy)
	}
	if len(policy.KeyUsages) == 0 {
		return nil, fmt.Errorf("%w: KeyUsages must contain at least one extended key usage", ErrInvalidPolicy)
	}

	hash := policy.Hash
	if hash == 0 {
		hash = crypto.SHA256
	}
	switch hash {
	case crypto.SHA256, crypto.SHA384, crypto.SHA512:
	default:
		return nil, fmt.Errorf("%w: unsupported hash %v, only SHA-256, SHA-384 and SHA-512 are allowed", ErrInvalidPolicy, hash)
	}
	if !hash.Available() {
		return nil, fmt.Errorf("%w: hash %v is not available", ErrInvalidPolicy, hash)
	}

	maxIntermediates := policy.MaxIntermediates
	if maxIntermediates == 0 {
		maxIntermediates = defaultMaxIntermediateCerts
	}
	if maxIntermediates < 1 || maxIntermediates > maxIntermediateCerts {
		return nil, fmt.Errorf("%w: MaxIntermediates %d must be between 1 and %d", ErrInvalidPolicy, maxIntermediates, maxIntermediateCerts)
	}

	if policy.ClockSkew < 0 || policy.ClockSkew > maxClockSkew {
		return nil, fmt.Errorf("%w: ClockSkew %s must be between 0 and %s", ErrInvalidPolicy, policy.ClockSkew, maxClockSkew)
	}

	switch policy.Mode {
	case TimeModeCurrent, TimeModeSigning:
	default:
		return nil, fmt.Errorf("%w: Mode %d is out of range", ErrInvalidPolicy, policy.Mode)
	}
	if policy.Mode == TimeModeSigning && policy.SigningTimeAuthority == nil {
		return nil, fmt.Errorf("%w: SigningTimeAuthority is required in signing-time mode", ErrInvalidPolicy)
	}

	return &CertificateVerifier{
		roots:             policy.Roots.Clone(),
		keyUsages:         append([]x509.ExtKeyUsage(nil), policy.KeyUsages...),
		hash:              hash,
		maxIntermediates:  maxIntermediates,
		clockSkew:         policy.ClockSkew,
		mode:              policy.Mode,
		signingTimeSource: policy.SigningTimeAuthority,
	}, nil
}

// Verify checks the signature of message against the leaf certificate and
// validates the certificate chain up to a configured trust root.
//
// Verification runs in fixed stages, returning at the first failure:
// input validation, PEM parsing, chain structure, signature verification,
// time resolution and validity, and finally trust-path construction. The
// supplied readers are never closed and the inputs are never modified.
//
// evaluationTime must be non-zero; ctx, message and signature must be
// non-nil. On failure the returned result is always nil. Context cancellation
// is checked at every stage boundary and takes precedence over later stages.
func (v *CertificateVerifier) Verify(ctx context.Context, leafPEM, intermediatesPEM []byte, message, signature io.Reader, evaluationTime time.Time) (*VerificationResult, error) {
	// Stage 1: input validation.
	if ctx == nil {
		return nil, fmt.Errorf("%w: context must not be nil", ErrInvalidInput)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if evaluationTime.IsZero() {
		return nil, fmt.Errorf("%w: evaluationTime must not be zero", ErrInvalidInput)
	}
	if message == nil {
		return nil, fmt.Errorf("%w: message reader must not be nil", ErrInvalidInput)
	}
	if signature == nil {
		return nil, fmt.Errorf("%w: signature reader must not be nil", ErrInvalidInput)
	}
	leafBlocks, leafCount, err := countCertificateBlocks(leafPEM)
	if err != nil {
		return nil, err
	}
	if leafCount != 1 {
		return nil, fmt.Errorf("%w: leaf PEM must contain exactly one certificate, found %d", ErrInvalidInput, leafCount)
	}

	// Stage 2: PEM parsing.
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	leafCerts, err := parseCertificateBlocks(leafBlocks)
	if err != nil {
		return nil, err
	}
	leaf := leafCerts[0]
	intermediateBlocks, intermediateCount, err := countCertificateBlocks(intermediatesPEM)
	if err != nil {
		return nil, err
	}
	intermediates, err := parseCertificateBlocks(intermediateBlocks)
	if err != nil {
		return nil, err
	}

	// Stage 3: chain structure.
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	chainCerts, roles, err := v.buildChainStructure(leaf, intermediates, intermediateCount)
	if err != nil {
		return nil, err
	}

	// Stage 4: signature verification.
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	messageDigest, signatureBytes, err := v.verifySignature(ctx, leaf, message, signature)
	if err != nil {
		return nil, err
	}

	// Stage 5: time resolution and certificate validity.
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	effectiveTime, signingTime, err := v.resolveEffectiveTime(ctx, leaf.Raw, messageDigest, signatureBytes, evaluationTime)
	if err != nil {
		return nil, err
	}
	if err := checkValidity(chainCerts, roles, effectiveTime); err != nil {
		return nil, err
	}

	// Stage 6: trust-path construction.
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	verifiedChain, err := v.verifyTrustPath(leaf, intermediates, effectiveTime)
	if err != nil {
		return nil, err
	}

	chainCopy := append([]*x509.Certificate(nil), verifiedChain...)
	roleCopy := make([]CertRole, len(chainCopy))
	for i := range chainCopy {
		switch {
		case i == 0:
			roleCopy[i] = RoleLeaf
		case i == len(chainCopy)-1:
			roleCopy[i] = RoleRoot
		default:
			roleCopy[i] = RoleIntermediate
		}
	}

	result := &VerificationResult{
		Leaf:          leaf,
		Chain:         chainCopy,
		Roles:         roleCopy,
		Mode:          v.mode,
		EffectiveTime: effectiveTime.UTC(),
		SigningTime:   signingTime,
	}
	return result, nil
}

// countCertificateBlocks walks the PEM blocks of data and returns the number
// of blocks carrying certificates. An empty input yields zero blocks without
// an error. Non-CERTIFICATE blocks, trailing non-PEM garbage and empty
// certificate blocks are reported as ErrMalformedCertificate.
func countCertificateBlocks(data []byte) ([]*pem.Block, int, error) {
	blocks := make([]*pem.Block, 0)
	remaining := bytes.TrimSpace(data)
	for len(remaining) > 0 {
		var block *pem.Block
		block, remaining = pem.Decode(remaining)
		if block == nil {
			return nil, 0, fmt.Errorf("%w: PEM decoding failed on trailing data", ErrMalformedCertificate)
		}
		if block.Type != string(cryptoutils.CertificatePEMType) {
			return nil, 0, fmt.Errorf("%w: expected PEM block type %q, got %q", ErrMalformedCertificate, cryptoutils.CertificatePEMType, block.Type)
		}
		if len(block.Bytes) == 0 {
			return nil, 0, fmt.Errorf("%w: empty PEM certificate block", ErrMalformedCertificate)
		}
		blocks = append(blocks, block)
		remaining = bytes.TrimSpace(remaining)
	}
	return blocks, len(blocks), nil
}

// parseCertificateBlocks parses the DER payloads of the blocks previously
// counted by countCertificateBlocks. Parse errors are retained and wrapped
// around ErrMalformedCertificate.
func parseCertificateBlocks(blocks []*pem.Block) ([]*x509.Certificate, error) {
	certs := make([]*x509.Certificate, 0, len(blocks))
	for _, block := range blocks {
		cert, err := x509.ParseCertificate(block.Bytes)
		if err != nil {
			return nil, fmt.Errorf("%w: %w", ErrMalformedCertificate, err)
		}
		if cert == nil {
			return nil, fmt.Errorf("%w: parsed certificate is nil", ErrMalformedCertificate)
		}
		certs = append(certs, cert)
	}
	return certs, nil
}

// buildChainStructure enforces the local chain rules that are independent of
// the x509 path verifier.
func (v *CertificateVerifier) buildChainStructure(leaf *x509.Certificate, intermediates []*x509.Certificate, intermediateCount int) ([]*x509.Certificate, []CertRole, error) {
	if leaf.IsCA {
		return nil, nil, ErrLeafIsCA
	}
	if intermediateCount > v.maxIntermediates {
		return nil, nil, fmt.Errorf("%w: %d intermediates exceed limit of %d", ErrTooManyIntermediates, intermediateCount, v.maxIntermediates)
	}

	chainCerts := []*x509.Certificate{leaf}
	roles := []CertRole{RoleLeaf}
	seenDER := map[string]struct{}{
		string(leaf.Raw): {},
	}
	previous := leaf
	for _, intermediate := range intermediates {
		if isSelfIssuedRoot(intermediate) {
			return nil, nil, fmt.Errorf("%w: root certificate %q must not be supplied as an intermediate", ErrInvalidChain, intermediate.Subject.String())
		}
		if !intermediate.IsCA {
			return nil, nil, fmt.Errorf("%w: intermediate certificate %q is not a CA", ErrInvalidChain, intermediate.Subject.String())
		}
		if _, duplicate := seenDER[string(intermediate.Raw)]; duplicate {
			return nil, nil, fmt.Errorf("%w: duplicate certificate %q in chain", ErrInvalidChain, intermediate.Subject.String())
		}
		// The intermediate is the issuer of the previous certificate: its
		// subject matches the previous issuer name, and the previous
		// certificate's signature verifies under the intermediate key.
		if !bytes.Equal(intermediate.RawSubject, previous.RawIssuer) {
			return nil, nil, fmt.Errorf("%w: certificate %q is not the issuer of %q", ErrInvalidChain, intermediate.Subject.String(), previous.Subject.String())
		}
		if err := previous.CheckSignatureFrom(intermediate); err != nil {
			return nil, nil, fmt.Errorf("%w: signature of %q does not verify under %q: %w", ErrInvalidChain, previous.Subject.String(), intermediate.Subject.String(), err)
		}
		seenDER[string(intermediate.Raw)] = struct{}{}
		chainCerts = append(chainCerts, intermediate)
		roles = append(roles, RoleIntermediate)
		previous = intermediate
	}
	return chainCerts, roles, nil
}

// isSelfIssuedRoot reports whether cert is a self-issued, self-signed
// certificate, i.e. a root trust anchor.
func isSelfIssuedRoot(cert *x509.Certificate) bool {
	if !bytes.Equal(cert.RawSubject, cert.RawIssuer) {
		return false
	}
	return cert.CheckSignatureFrom(cert) == nil
}

// verifySignature reads the full message and signature, checks the signature
// size limit, loads a verifier for the leaf key and verifies the signature.
// It returns the SHA-256 digest of the complete message and the signature
// bytes, both needed by the signing-time authority.
func (v *CertificateVerifier) verifySignature(_ context.Context, leaf *x509.Certificate, message, signature io.Reader) ([32]byte, []byte, error) {
	var digest [32]byte
	// Buffer the message so it can be replayed to the verifier while its
	// complete byte stream is also hashed for the signing-time request.
	messageBytes, err := io.ReadAll(message)
	if err != nil {
		return digest, nil, fmt.Errorf("%w: reading message: %w", ErrSignatureInvalid, err)
	}
	digest = sha256.Sum256(messageBytes)

	// Read at most maxSignatureBytes plus one byte so that an oversized
	// signature is detected rather than silently truncated.
	signatureBytes, err := io.ReadAll(io.LimitReader(signature, maxSignatureBytes+1))
	if err != nil {
		return digest, nil, fmt.Errorf("%w: reading signature: %w", ErrSignatureInvalid, err)
	}
	if len(signatureBytes) > maxSignatureBytes {
		return digest, nil, fmt.Errorf("%w: signature is larger than %d bytes", ErrSignatureTooLarge, maxSignatureBytes)
	}

	verifier, err := LoadVerifierWithOpts(leaf.PublicKey, options.WithHash(v.hash))
	if err != nil {
		return digest, nil, fmt.Errorf("%w: %w", ErrUnsupportedKey, err)
	}
	if err := verifier.VerifySignature(bytes.NewReader(signatureBytes), bytes.NewReader(messageBytes)); err != nil {
		return digest, nil, fmt.Errorf("%w: %w", ErrSignatureInvalid, err)
	}
	return digest, signatureBytes, nil
}

// resolveEffectiveTime implements the mode-specific time semantics. In
// current mode the evaluation time is used directly and the authority is
// never called. In signing mode the authority is invoked exactly once; its
// result is rejected when it is zero, erroneous or in the future beyond the
// clock skew.
func (v *CertificateVerifier) resolveEffectiveTime(ctx context.Context, leafDER []byte, messageDigest [32]byte, signatureBytes []byte, evaluationTime time.Time) (time.Time, *time.Time, error) {
	if v.mode == TimeModeCurrent {
		return evaluationTime, nil, nil
	}

	leafDERCopy := append([]byte(nil), leafDER...)
	signatureCopy := append([]byte(nil), signatureBytes...)
	request := SigningTimeRequest{
		LeafDER:       leafDERCopy,
		MessageDigest: messageDigest,
		Signature:     signatureCopy,
	}
	signingTime, err := v.signingTimeSource(ctx, request)
	if err != nil {
		return time.Time{}, nil, fmt.Errorf("%w: %w", ErrSigningTimeUnavailable, err)
	}
	// Cancellation observed after the authority returns takes precedence and
	// must never fall back to evaluation-time semantics.
	if ctxErr := ctx.Err(); ctxErr != nil {
		return time.Time{}, nil, ctxErr
	}
	if signingTime.IsZero() {
		return time.Time{}, nil, fmt.Errorf("%w: authority returned zero time", ErrSigningTimeUnavailable)
	}
	if signingTime.After(evaluationTime.Add(v.clockSkew)) {
		return time.Time{}, nil, fmt.Errorf("%w: signing time %s is after evaluation time %s plus clock skew %s",
			ErrSigningTimeInFuture, signingTime.UTC().Format(time.RFC3339), evaluationTime.UTC().Format(time.RFC3339), v.clockSkew)
	}
	signingTimeUTC := signingTime.UTC()
	return signingTimeUTC, &signingTimeUTC, nil
}

// checkValidity enforces the certificate validity windows in the order leaf,
// intermediates as supplied, then root. Equality with NotBefore or NotAfter
// is accepted.
func checkValidity(certs []*x509.Certificate, roles []CertRole, effectiveTime time.Time) error {
	for i, cert := range certs {
		role := roles[i]
		if effectiveTime.Before(cert.NotBefore) {
			return &CertificateValidityError{
				Role:          role,
				Subject:       cert.Subject.String(),
				NotBefore:     cert.NotBefore.UTC(),
				NotAfter:      cert.NotAfter.UTC(),
				EffectiveTime: effectiveTime.UTC(),
				Expired:       false,
			}
		}
		if effectiveTime.After(cert.NotAfter) {
			return &CertificateValidityError{
				Role:          role,
				Subject:       cert.Subject.String(),
				NotBefore:     cert.NotBefore.UTC(),
				NotAfter:      cert.NotAfter.UTC(),
				EffectiveTime: effectiveTime.UTC(),
				Expired:       true,
			}
		}
	}
	return nil
}

// verifyTrustPath builds the x509 path to a trusted root. Local pools are
// created per call so the verifier's internal root pool is never written to.
func (v *CertificateVerifier) verifyTrustPath(leaf *x509.Certificate, intermediates []*x509.Certificate, effectiveTime time.Time) ([]*x509.Certificate, error) {
	intermediatePool := x509.NewCertPool()
	for _, intermediate := range intermediates {
		intermediatePool.AddCert(intermediate)
	}
	chains, err := leaf.Verify(x509.VerifyOptions{
		Roots:         v.roots,
		Intermediates: intermediatePool,
		CurrentTime:   effectiveTime,
		KeyUsages:     v.keyUsages,
	})
	if err != nil {
		return nil, v.mapTrustError(err, effectiveTime)
	}
	return chains[0], nil
}

// mapTrustError translates x509 verification failures. Expiry reported by the
// path verifier can only originate from the root (leaf and intermediates are
// checked explicitly beforehand) and is surfaced as a
// CertificateValidityError with RoleRoot rather than being wrapped in
// ErrUntrustedChain. Every other failure is wrapped around
// ErrUntrustedChain, and the original x509 error stays reachable via
// errors.As.
func (v *CertificateVerifier) mapTrustError(err error, effectiveTime time.Time) error {
	var invalid x509.CertificateInvalidError
	if errors.As(err, &invalid) && invalid.Cert != nil &&
		(invalid.Reason == x509.Expired ||
			effectiveTime.After(invalid.Cert.NotAfter) ||
			effectiveTime.Before(invalid.Cert.NotBefore)) {
		return &CertificateValidityError{
			Role:          RoleRoot,
			Subject:       invalid.Cert.Subject.String(),
			NotBefore:     invalid.Cert.NotBefore.UTC(),
			NotAfter:      invalid.Cert.NotAfter.UTC(),
			EffectiveTime: effectiveTime.UTC(),
			Expired:       effectiveTime.After(invalid.Cert.NotAfter),
		}
	}
	return fmt.Errorf("%w: %w", ErrUntrustedChain, err)
}
