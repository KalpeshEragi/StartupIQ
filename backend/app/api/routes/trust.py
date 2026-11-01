"""The verification report: can these numbers be trusted, and how was each one made? Printable for investors."""
from collections import defaultdict

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import Access, startup_access
from app.core.db import get_db
from app.models import BankAccount, CategorySource, StatementImport, Transaction, User
from app.models.base import utcnow
from app.schemas import AuditChainOut, ImportCheckOut, TrustCounts, TrustMonth, VerificationOut
from app.services import audit, verification

router = APIRouter(prefix="/startups/{startup_id}", tags=["verification"])

LIMITS = ("These checks catch edited, missing or reordered lines, and statements missing between uploads. They cannot "
          "prove a file came from the bank, because a file can be edited before it is uploaded; only bank-signed data "
          "can prove that.")
LEGACY = "Uploaded before statement checks existed. Upload the same file again to check it."
AUTO = (CategorySource.memory, CategorySource.rule, CategorySource.model)


def _counts(rows: list[tuple]) -> dict:
    """rows: (needs_review, source, category, auto_category, import_status)."""
    return {"total": len(rows), "verified": sum(s == "verified" for *_, s in rows),
            "auto_filed": sum(not r and src in AUTO for r, src, *_ in rows),
            "founder_set": sum(src == CategorySource.founder for _, src, *_ in rows),
            "changed": sum(src == CategorySource.founder and a is not None and c != a for _, src, c, a, _ in rows),
            "needs_review": sum(r for r, *_ in rows)}


@router.get("/verification", response_model=VerificationOut)
def report(access: Access = Depends(startup_access()), db: Session = Depends(get_db)) -> VerificationOut:
    sid = access.startup.id
    imports = db.execute(select(StatementImport, BankAccount.name, User.full_name)
                         .join(BankAccount, BankAccount.id == StatementImport.bank_account_id)
                         .outerjoin(User, User.id == StatementImport.uploaded_by)
                         .where(StatementImport.startup_id == sid).order_by(StatementImport.created_at)).all()
    checks = [ImportCheckOut(id=i.id, filename=i.filename, account=account, uploaded_at=i.created_at, uploaded_by=who,
                             period_start=i.period_start, period_end=i.period_end, rows_added=i.rows_added,
                             file_sha256=i.file_sha256, status=i.balance_status or "unverified",
                             balance_checked=i.balance_checked or 0, balance_breaks=i.balance_breaks or 0,
                             opening_balance=i.opening_balance, closing_balance=i.closing_balance,
                             issues=(i.issues or []) if i.balance_status else [LEGACY])
              for i, account, who in imports]

    by_account: dict = defaultdict(list)
    for i, _, _ in imports:
        by_account[i.bank_account_id].append(verification.StatementSpan(i.filename, i.period_start, i.period_end,
                                                                        i.opening_balance, i.closing_balance))
    gaps = [issue for spans in by_account.values() for issue in verification.continuity(spans)]

    rows = db.execute(select(Transaction.txn_date, Transaction.needs_review, Transaction.category_source, Transaction.category,
                             Transaction.auto_category, Transaction.import_id, Transaction.fingerprint, Transaction.direction,
                             Transaction.amount, Transaction.balance, Transaction.description)
                      .where(Transaction.startup_id == sid)).all()
    stored: dict = defaultdict(list)
    for r in rows:
        stored[r.import_id].append((r.fingerprint, r.txn_date, str(r.direction), r.amount, r.balance, r.description))
    tampered = {i.id for i, _, _ in imports if i.rows_sha256 and verification.rows_seal(stored[i.id]) != i.rows_sha256}
    for check in checks:
        if check.id in tampered:
            check.status = "warnings"
            check.issues.insert(0, "Lines from this statement were changed or deleted after upload, outside the app.")
    status_of = {c.id: c.status for c in checks}
    lines = [(r.needs_review, r.category_source, r.category, r.auto_category, status_of.get(r.import_id)) for r in rows]
    months: dict[str, list] = defaultdict(list)
    for r, line in zip(rows, lines, strict=True):
        months[f"{r.txn_date:%Y-%m}"].append(line)

    chain = audit.verify_chain(db, sid)
    statuses = {c.status for c in checks}
    broken = sum(c.balance_breaks for c in checks)
    if not checks:
        status, headline = "unverified", "No statements uploaded yet."
    elif "warnings" in statuses or gaps or not chain.ok:
        status = "warnings"
        problems = [f"{len(tampered)} statement(s) changed after upload" if tampered else "",
                    f"{broken} line(s) whose balance does not add up" if broken else "",
                    f"{len(gaps)} possible missing statement(s)" if gaps else "",
                    "the activity history was changed" if not chain.ok else ""]
        found = ", ".join(p for p in problems if p)
        headline = f"Found {found}."
    elif statuses == {"verified"}:
        status, headline = "verified", "Every statement adds up, and the activity history is intact."
    elif any(i.balance_status is None for i, _, _ in imports):
        status, headline = "unverified", "Some statements were uploaded before checks existed. Upload them again to check them."
    else:
        status, headline = "unverified", "Some statements have no balance column, so they could not be checked."

    return VerificationOut(generated_at=utcnow(), status=status, headline=headline, lines=TrustCounts(**_counts(lines)),
                           months=[TrustMonth(month=m, **_counts(v)) for m, v in sorted(months.items(), reverse=True)],
                           imports=checks, continuity=gaps, limits=LIMITS,
                           audit=AuditChainOut(ok=chain.ok, sealed=chain.sealed, unsealed=chain.unsealed, head=chain.head,
                                               problem=chain.problem))
