"""
FraudSentinel — ChromaDB vector store population (RAG context for fraud explanations).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


CHROMA_DIR_NAME = "chroma_db"
COLLECTION_NAME = "fraud_cases"


def _project_root() -> Path:
    return Path(__file__).resolve().parent


class DeterministicEmbeddingFunction(EmbeddingFunction[Documents]):
    """L2-normalized pseudo-embeddings (no model download) for offline-friendly demos."""

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed_one(text) for text in input]

    def _embed_one(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little", signed=False) % (2**32 - 1) + 1
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self._dimension, dtype=np.float64)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32).tolist()


def build_chroma_client(persist_directory: Path) -> chromadb.PersistentClient:
    """Create a persistent Chroma client rooted at persist_directory."""
    persist_directory.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_directory))


def create_fraud_cases_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """(Re)create fraud_cases with a deterministic embedding function so no ONNX download is required."""
    existing = {c.name for c in client.list_collections()}
    if COLLECTION_NAME in existing:
        # Collection exists — check if it has documents
        coll = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=DeterministicEmbeddingFunction(),
        )
        try:
            count = coll.count()
            if count > 0:
                print(f"[RAG] Collection '{COLLECTION_NAME}' already exists with {count} documents — skipping recreation")
                return coll
        except Exception:
            pass
        # Collection exists but is empty — delete and recreate
        client.delete_collection(COLLECTION_NAME)
    
    return client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Synthetic fraud scenarios for RAG"},
        embedding_function=DeterministicEmbeddingFunction(),
    )


def synthetic_fraud_cases() -> list[dict[str, Any]]:
    """Fifty realistic 2–3 sentence fraud scenarios: five per fraud type."""
    types_and_cases: list[tuple[str, list[str]]] = [
        (
            "velocity_attack",
            [
                "In under four minutes the card was used at twelve different gas stations for sub-twenty-dollar fuel purchases across two counties. "
                "The pattern matches scripted bots testing which authorizations clear before a larger push. "
                "The legitimate cardholder had not left home that morning per device telemetry.",
                "Seventeen contactless taps under fifteen dollars each occurred at fast-food terminals along a single highway corridor within twenty minutes. "
                "No matching travel was logged on the issuer’s mobile app. "
                "Velocity limits were bypassed because each merchant used a different acquirer batch window.",
                "A dormant card suddenly registered forty-two micro-charges at online marketplaces in ninety minutes, each just below the issuer’s SMS alert threshold. "
                "Shipping addresses rotated through freight-forwarding hubs. "
                "The account had seen fewer than five transactions in the prior year.",
                "ATM withdrawals hit the daily limit in six back-to-back transactions at machines spaced minutes apart on foot. "
                "PIN entry timing was unnaturally consistent across terminals. "
                "The card was reported in a wallet at the customer’s office during the same window.",
                "Rapid-fire digital wallet authorizations at coffee chains cycled through tokenized PAN variants tied to the same underlying account. "
                "Each charge was small enough to avoid step-up authentication. "
                "The burst ended with a failed attempt at a luxury jeweler that triggered review.",
            ],
        ),
        (
            "geographic_anomaly",
            [
                "A grocery purchase cleared in Seattle while the same card presented for a jewelry purchase in Miami twelve minutes later. "
                "Neither city matched the customer’s home ZIP or recent travel history. "
                "The jewelry merchant had no prior relationship with the card.",
                "Chip-and-PIN dinner in London overlapped with an online electronics order shipping to Eastern Europe using the same credentials. "
                "The customer had no flight bookings or roaming on file. "
                "3-D Secure was not invoked because the e-commerce merchant was exempt.",
                "Contactless transit taps in Tokyo appeared while mobile banking showed the handset geolocated in rural Texas. "
                "VPN use was not evident on the session. "
                "The transit pattern matched tourist routes rather than the cardholder’s commute.",
                "A fuel pump authorization in rural Montana was followed by a high-end handbag purchase in Manhattan within an impossible drive time. "
                "Both merchants posted as card-present. "
                "The issuer’s overnight batch flagged no travel notice on file.",
                "Hotel incidentals in Dubai posted while payroll direct deposit showed the employee clocked in at a Midwest distribution center. "
                "Corporate travel policy had no approved trip. "
                "The hotel folio included spa charges typical of third-party booking fraud.",
            ],
        ),
        (
            "merchant_mismatch",
            [
                "A student meal-plan branded card suddenly funded a five-figure purchase at a high-end watch dealer. "
                "Merchant category code conflicted with the product type on the receipt. "
                "Prior six months of spend were exclusively campus dining and textbooks.",
                "A corporate procurement card coded to office supplies was used for cryptocurrency gift cards at a big-box retailer. "
                "MCC suggested general merchandise while line items implied digital assets. "
                "The employee role had no treasury privileges.",
                "A grocery loyalty co-brand showed repeated charges at an online gambling platform disguised as digital goods. "
                "Statement descriptors were obfuscated through payment aggregators. "
                "Velocity was low but ticket size was high versus historical basket.",
                "A transit-only prepaid wallet was drained through a payment facilitator routing to luxury apparel. "
                "The facilitator MCC differed from downstream settlement. "
                "Device fingerprint matched a known fraud ring browser profile.",
                "A healthcare FSA card attempted payment at a motorsports dealership for performance parts. "
                "Merchant name normalization masked the mismatch until settlement. "
                "Benefit rules prohibit non-medical spend.",
            ],
        ),
        (
            "late_night_high_value",
            [
                "At 2:14 a.m. local time, a single swipe cleared for a full-limit jewelry purchase while the cardholder’s phone showed do-not-disturb hours. "
                "No prior late-night spend existed on the account. "
                "The clerk manually keyed the PAN after chip read failures.",
                "Between midnight and 3 a.m., three consecutive wire-card loads hit the daily funding cap at kiosks in a casino district. "
                "Each load was immediately followed by ATM withdrawal attempts. "
                "Home ZIP was three time zones away with no flight correlation.",
                "A dormant premium card authorized a six-thousand-dollar home electronics cart at 1:50 a.m. with ship-to-store pickup under a generic name. "
                "Billing and shipping ZIPs diverged from on-file addresses. "
                "Step-up SMS never arrived due to ported number flags.",
                "Overnight, a business card paid for charter aircraft catering and ramp fees in two countries. "
                "No corporate travel request existed for those tail numbers. "
                "Card was physically held in a locked office safe per HR attestation.",
                "At 3:40 a.m., a contactless hotel folio posted a presidential suite upgrade and minibar clear-out. "
                "Guest name on folio did not match the card emboss. "
                "Front desk camera later showed a different individual presenting the card.",
            ],
        ),
        (
            "card_not_present_new_device",
            [
                "A first-time browser with no cookies completed a card-not-present electronics checkout using saved credentials from a password dump. "
                "Device fingerprint showed Linux headless automation. "
                "IP reputation was poor and geolocation mismatched billing ZIP.",
                "New mobile device ID paired with an email change request minutes before a digital gift card binge. "
                "No prior logins from that handset family existed. "
                "3-D Secure frictionless path was taken due to low-risk scoring misconfiguration.",
                "A CNP subscription stack signed up for nine streaming trials using the same PAN with rotating disposable emails. "
                "TLS fingerprint indicated datacenter egress. "
                "BIN country differed from IP country without VPN indicators.",
                "Wallet provisioning from an unfamiliar OEM handset preceded immediate peer-to-peer loads from the same card token. "
                "Biometric enrollment on file was never completed for that device. "
                "Issuer risk engine had not refreshed device reputation cache.",
                "Browser language and timezone skewed heavily from cardholder profile during a late-night digital goods sweep. "
                "Keyboard cadence analysis suggested scripted entry. "
                "Shipping address was a vacant-lot freight forwarder.",
            ],
        ),
        (
            "sim_swap_transfer",
            [
                "Customer service logs show a SIM swap completed at retail; within minutes, one-time passcodes were intercepted and a wire transfer was initiated. "
                "The beneficiary account was opened days earlier with minimal KYC. "
                "The handset IMEI changed while the customer still had possession of the original phone hardware.",
                "Account recovery codes were reset after carrier port-out approval; attackers drained investment balances to stablecoin off-ramps. "
                "SMS-based 2FA was the only second factor on file. "
                "Geolocation of banking sessions jumped continents within an hour.",
                "A prepaid SIM activation in another state preceded password resets on email and banking within the same hour. "
                "OTP delivery shifted to the attacker-controlled handset. "
                "Legitimate customer was traveling without international roaming, delaying awareness.",
                "Port validation questions were answered using data leaked from prior breaches; SIM swap triggered instant P2P limits to max. "
                "Outbound transfers used mule names rotating through the same routing number. "
                "Carrier notes showed social engineering on support line.",
                "Dual-SIM fraud: secondary line activated on stolen ID, then primary number hijacked for banking OTP. "
                "Large international remittance posted before fraud desk callback window. "
                "Customer first learned when debit card declined at grocery.",
            ],
        ),
        (
            "micro_test_large_fraud",
            [
                "Two one-dollar authorizations to digital merchants succeeded overnight; the next day a fifteen-thousand-dollar jewelry authorization cleared before alerts fired. "
                "The micro-tests used different merchant IDs under the same processor. "
                "Card had been inactive for months prior.",
                "Penny verification charges posted to ride-hail and food delivery, then a consolidated invoice merchant billed the full credit line. "
                "Each micro-test used a different BIN range tokenization path. "
                "Fraud team later tied the invoice merchant to laundering shell.",
                "Sub-dollar charity donations validated PAN validity across three continents within an hour. "
                "A wire-on-card product then moved the majority of available credit. "
                "MCCs for charities were intentionally chosen to avoid scrutiny.",
                "Five-cent card validation hits from SaaS trials escalated to bulk electronics purchase using stored credentials. "
                "Trials never converted to paid plans under legitimate email. "
                "Shipping consolidated to a reshipper address.",
                "Sequential $0.50 music-store purchases tested AVS responses; once match codes stabilized, a high-ticket auction win was paid. "
                "Auction account age was under twenty-four hours. "
                "Seller payout went to a prepaid debit product.",
            ],
        ),
        (
            "refund_fraud",
            [
                "A merchant processed a refund to a different card ending than the original purchase PAN, citing customer request over chat. "
                "Original sale was legitimate; refund landed on a stolen card on file. "
                "Chargeback on original followed, leaving the merchant double-hit.",
                "Serial partial refunds on high-ticket electronics exceeded original capture due to rounding and fee manipulation across channels. "
                "Refund destinations cycled through prepaid products. "
                "Customer history showed pattern across multiple retailers.",
                "Friendly-fraud claimant insisted non-receipt while tracking showed delivery; simultaneous refund-to-original and chargeback were attempted. "
                "Issuer notes referenced conflicting narratives. "
                "Delivery photo included different individual signing.",
                "Refund abuse on digital goods: customer consumed license keys then claimed accidental purchase for full refund. "
                "Keys were redeemed from IPs tied to resale marketplaces. "
                "Same pattern repeated across vendor accounts.",
                "Split-tender purchase followed by refund request to cash-equivalent gift card only, bypassing original card reversal rules. "
                "Store policy exception was granted under pressure. "
                "Surveillance showed collusion with cashier.",
            ],
        ),
        (
            "cross_border_no_travel",
            [
                "Multiple card-present taps in a Schengen hub posted while passport and travel-benefit records showed no trip. "
                "Issuer travel notice mailbox was empty. "
                "Merchant receipts showed signatures inconsistent with on-file specimen.",
                "Cross-border e-commerce in three currencies within a day with ship-to addresses in high-risk corridors. "
                "Cardholder mobile app last login remained domestic. "
                "No roaming charges on telecom file match.",
                "ATM withdrawals in two South American cities while payroll ACH showed regular domestic direct deposit activity. "
                "No airline ticket or hotel folio matched dates. "
                "Skimming overlay later found at a domestic gas pump used days earlier.",
                "Luxury hotel and car rental in Asia billed while physical card was replaced domestically for chip damage—old PAN still active on magstripe fallback. "
                "Travel insurance rider was not triggered. "
                "Magstripe fallback occurred at merchants with weak chip enforcement.",
                "International wire-on-card loads from EU acquirers without corresponding FX travel alerts on the profile. "
                "Customer occupation listed as local government employee with no overseas duty. "
                "BIN-level controls expected travel flag that never arrived.",
            ],
        ),
        (
            "subscription_chargeback",
            [
                "Subscriber signed up for annual SaaS, consumed API quotas heavily for thirty days, then filed chargeback claiming unauthorized renewal. "
                "Login logs showed credential use from customer home IP throughout. "
                "Support tickets praised the product weeks before dispute.",
                "Trial-to-paid conversion used stolen card; after service delivery, chargeback cited not-as-described despite Terms acceptance timestamps. "
                "Digital delivery logs proved usage. "
                "Issuer sided with customer under consumer protection rules.",
                "Gaming subscription stacked with in-app purchases; chargebacks rolled in after chargeback rights window manipulation across processors. "
                "Device matched known friendly-fraud cohort. "
                "Refunds were refused per policy then bank dispute opened.",
                "Fitness app bundled hardware shipped; user initiated chargeback on hardware while retaining device per resale listings. "
                "Serial number matched shipment. "
                "Social posts showed unboxing same week.",
                "Streaming bundle with annual discount was charged back after password sharing monetization; multiple households used credentials simultaneously. "
                "Fraud team flagged reseller forum posts selling shared logins. "
                "Original subscriber denied knowing co-users.",
            ],
        ),
    ]

    cases: list[dict[str, Any]] = []
    n = 0
    for fraud_type, texts in types_and_cases:
        for text in texts:
            n += 1
            cases.append(
                {
                    "id": f"case_{n:03d}",
                    "type": fraud_type,
                    "document": text.strip(),
                }
            )
    assert len(cases) == 50, len(cases)
    return cases


def populate_fraud_cases_collection(
    collection: chromadb.Collection,
    cases: list[dict[str, Any]],
) -> None:
    """Upsert synthetic fraud documents with metadata into the collection."""
    ids = [c["id"] for c in cases]
    documents = [c["document"] for c in cases]
    metadatas = [{"type": c["type"], "id": c["id"]} for c in cases]
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)


def run_smoke_query(collection: chromadb.Collection, query: str, k: int = 3) -> None:
    """Query the collection and print the top k hits (ids, distances, metadata, document snippet)."""
    result = collection.query(query_texts=[query], n_results=k)
    ids = result["ids"][0] if result.get("ids") else []
    dists = result["distances"][0] if result.get("distances") else []
    docs = result["documents"][0] if result.get("documents") else []
    metas = result["metadatas"][0] if result.get("metadatas") else []
    print(f'Query: "{query}" - top {k} results:')
    for rank, (cid, dist, doc, meta) in enumerate(
        zip(ids, dists, docs, metas, strict=True), start=1
    ):
        snippet = (doc or "")[:220] + ("…" if doc and len(doc) > 220 else "")
        print(f"  {rank}. id={cid} distance={dist} metadata={meta}")
        print(f"     {snippet}")


def main() -> None:
    root = _project_root()
    persist = root / CHROMA_DIR_NAME
    client = build_chroma_client(persist)
    collection = create_fraud_cases_collection(client)
    cases = synthetic_fraud_cases()
    populate_fraud_cases_collection(collection, cases)
    print("RAG setup complete: 50 cases indexed")
    run_smoke_query(collection, "unusual location high amount", k=3)


if __name__ == "__main__":
    main()
