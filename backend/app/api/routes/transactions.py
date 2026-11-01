"""Bank accounts, statement upload, transactions and the review inbox. Every route is scoped to one startup."""
import re
import uuid
from collections import Counter
from datetime import date

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.api.deps import Access, startup_access
from app.categorisation import service
from app.categorisation.categories import ALL_CATEGORIES, EXPENSE_CATEGORIES, MANUAL_CATEGORIES
from app.categorisation.model import load_model
from app.categorisation.statements import StatementError, parse_statement
from app.core.db import get_db
from app.models import BankAccount, CategoryCorrection, CategorySource, StatementImport, Transaction, User
from app.schemas import (
    BankAccountIn,
    BankAccountOut,
    CategoryOut,
    CorrectionIn,
    CorrectionOut,
    HistoryEvent,
    ImportOut,
    ReviewSummary,
    TransactionHistory,
    TransactionOut,
    TransactionPage,
)
from app.services import audit, verification

router = APIRouter(prefix="/startups/{startup_id}", tags=["transactions"])
MAX_UPLOAD = 10 * 1024 * 1024
ALLOWED = (".csv", ".txt", ".xlsx", ".xls", ".xlsm")


@router.get("/categories", response_model=list[CategoryOut])
def categories(access: Access = Depends(startup_access())) -> list[CategoryOut]:
    expense = EXPENSE_CATEGORIES | MANUAL_CATEGORIES
    return [CategoryOut(key=k, label=v, group="expense" if k in expense else "structural")
            for k, v in ALL_CATEGORIES.items()]


@router.get("/bank-accounts", response_model=list[BankAccountOut])
def list_accounts(access: Access = Depends(startup_access()), db: Session = Depends(get_db)) -> list[BankAccount]:
    query = select(BankAccount).where(BankAccount.startup_id == access.startup.id).order_by(BankAccount.created_at)
    return list(db.scalars(query))


@router.post("/bank-accounts", response_model=BankAccountOut, status_code=status.HTTP_201_CREATED)
def create_account(body: BankAccountIn, request: Request, access: Access = Depends(startup_access()),
                   db: Session = Depends(get_db)) -> BankAccount:
    account = BankAccount(startup_id=access.startup.id, **body.model_dump())
    db.add(account)
    db.flush()
    audit.record(db, "bank_account.create", user_id=access.user.id, startup_id=access.startup.id, entity_id=account.id,
                 details={"name": account.name}, request=request)
    db.commit()
    return account


def _account(db: Session, access: Access, account_id: uuid.UUID) -> BankAccount:
    account = db.get(BankAccount, account_id)
    if account is None or account.startup_id != access.startup.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Bank account not found")
    return account


@router.post("/bank-accounts/{account_id}/imports", response_model=ImportOut, status_code=status.HTTP_201_CREATED)
async def import_statement(account_id: uuid.UUID, request: Request, file: UploadFile = File(...),
                           access: Access = Depends(startup_access()), db: Session = Depends(get_db)) -> ImportOut:
    account = _account(db, access, account_id)
    name = re.split(r"[\\/]", file.filename or "")[-1][:255] or "statement.csv"  # the name only, never a path
    if not name.lower().endswith(ALLOWED):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Upload a CSV or Excel statement.")
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "The file is larger than 10 MB.")
    try:
        parsed = parse_statement(data, name)
    except StatementError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from None

    sha = verification.file_sha256(data)
    same_file = db.scalar(select(StatementImport.filename).where(StatementImport.bank_account_id == account.id,
                                                                 StatementImport.file_sha256 == sha).limit(1))
    check = verification.check_balances(parsed.rows)
    prints = [r.fingerprint for r in parsed.rows]
    existing: set[str] = set()
    for i in range(0, len(prints), 1000):  # databases cap the values in one query (SQLite: 32,766)
        existing.update(db.scalars(select(Transaction.fingerprint).where(Transaction.bank_account_id == account.id,
                                                                         Transaction.fingerprint.in_(prints[i:i + 1000]))))
    new_rows = [r for r in parsed.rows if r.fingerprint not in existing]
    imp = StatementImport(startup_id=access.startup.id, bank_account_id=account.id, filename=name, uploaded_by=access.user.id,
                          rows_read=len(parsed.rows) + parsed.skipped, rows_added=len(new_rows),
                          rows_duplicate=len(parsed.rows) - len(new_rows), rows_skipped=parsed.skipped,
                          period_start=min(r.txn_date for r in parsed.rows), period_end=max(r.txn_date for r in parsed.rows),
                          file_sha256=sha, balance_status=check.status, balance_checked=check.checked,
                          balance_breaks=check.breaks, opening_balance=check.opening, closing_balance=check.closing,
                          issues=check.issues)
    db.add(imp)
    db.flush()
    _recheck_older_imports(db, account, parsed.rows, imp)
    txns = [Transaction(startup_id=access.startup.id, bank_account_id=account.id, import_id=imp.id, txn_date=r.txn_date,
                        description=r.description, direction=r.direction, amount=r.amount, balance=r.balance,
                        fingerprint=r.fingerprint, vendor_key=r.vendor_key, category_source=CategorySource.none,
                        source_row=r.line)
            for r in new_rows]
    service.categorise(db, access.startup.id, txns)
    for t in txns:
        t.auto_category, t.auto_source, t.auto_reason = t.category, str(t.category_source), t.reason
    imp.rows_sha256 = verification.rows_seal([(t.fingerprint, t.txn_date, str(t.direction), t.amount, t.balance, t.description)
                                              for t in txns])
    db.add_all(txns)
    audit.record(db, "statement.import", user_id=access.user.id, startup_id=access.startup.id, entity_id=imp.id,
                 details={"file": name, "sha256": sha, "rows_sha256": imp.rows_sha256, "added": imp.rows_added,
                          "duplicates": imp.rows_duplicate, "balance_status": check.status}, request=request)
    db.commit()
    return ImportOut(id=imp.id, filename=name, rows_read=imp.rows_read, rows_added=imp.rows_added,
                     rows_duplicate=imp.rows_duplicate, rows_skipped=imp.rows_skipped, period_start=imp.period_start,
                     period_end=imp.period_end, columns=parsed.columns, notes=parsed.notes,
                     auto_filed=sum(not t.needs_review for t in txns), needs_review=sum(t.needs_review for t in txns),
                     by_source=dict(Counter(str(t.category_source) for t in txns)), file_sha256=sha,
                     balance_status=check.status, balance_checked=check.checked, balance_breaks=check.breaks,
                     issues=check.issues, same_file_as=same_file)


def _recheck_older_imports(db: Session, account: BankAccount, rows: list, imp: StatementImport) -> None:
    """Imports from before statement checks existed get this upload's result when every line they added is in this
    file, so uploading the same statement again is how a founder gets an old statement checked."""
    in_file = {r.fingerprint for r in rows}
    for old in db.scalars(select(StatementImport).where(StatementImport.bank_account_id == account.id,
                                                        StatementImport.balance_status.is_(None))):
        lines = db.execute(select(Transaction.fingerprint, Transaction.txn_date, Transaction.direction, Transaction.amount,
                                  Transaction.balance, Transaction.description).where(Transaction.import_id == old.id)).all()
        if not lines or not {line.fingerprint for line in lines} <= in_file:
            continue
        old.file_sha256, old.balance_status = imp.file_sha256, imp.balance_status
        old.balance_checked, old.balance_breaks = imp.balance_checked, imp.balance_breaks
        old.opening_balance, old.closing_balance = imp.opening_balance, imp.closing_balance
        old.issues = [f"Checked with {imp.filename}, uploaded later.", *(imp.issues or [])]
        old.rows_sha256 = verification.rows_seal([(f, d, str(dr), a, b, desc) for f, d, dr, a, b, desc in lines])


@router.get("/transactions", response_model=TransactionPage)
def list_transactions(access: Access = Depends(startup_access()), db: Session = Depends(get_db),
                      needs_review: bool | None = None, category: str | None = None, q: str | None = Query(None, max_length=100),
                      month: str | None = Query(None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
                      limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)) -> TransactionPage:
    where = [Transaction.startup_id == access.startup.id]
    if needs_review is not None:
        where.append(Transaction.needs_review.is_(needs_review))
    if category:
        where.append(Transaction.category.is_(None) if category == "uncategorised" else Transaction.category == category)
    if q:
        like = f"%{q.lower()}%"
        where.append(or_(func.lower(Transaction.description).like(like), Transaction.vendor_key.like(like)))
    if month:
        y, m = map(int, month.split("-"))
        start, end = date(y, m, 1), date(y + (m == 12), m % 12 + 1, 1)  # dates, not strings: PostgreSQL will not compare them
        where += [Transaction.txn_date >= start, Transaction.txn_date < end]
    total = db.scalar(select(func.count()).select_from(Transaction).where(*where)) or 0
    items = db.scalars(select(Transaction).where(*where).order_by(Transaction.txn_date.desc(), Transaction.created_at.desc())
                       .limit(limit).offset(offset))
    return TransactionPage(items=[TransactionOut.model_validate(t) for t in items], total=total)


@router.get("/transactions/summary", response_model=ReviewSummary)
def summary(access: Access = Depends(startup_access()), db: Session = Depends(get_db)) -> ReviewSummary:
    rows = db.execute(select(Transaction.category_source, Transaction.needs_review, func.count())
                      .where(Transaction.startup_id == access.startup.id)
                      .group_by(Transaction.category_source, Transaction.needs_review)).all()
    total = sum(n for _, _, n in rows)
    review = sum(n for _, r, n in rows if r)
    by_source: Counter[str] = Counter()
    for source, _, n in rows:
        by_source[str(source)] += n
    model = load_model()
    return ReviewSummary(total=total, needs_review=review, auto_filed_share=(total - review) / total if total else 0.0,
                         by_source=dict(by_source), model_version=model.version if model else None)


def _transaction(db: Session, access: Access, txn_id: uuid.UUID) -> Transaction:
    txn = db.get(Transaction, txn_id)
    if txn is None or txn.startup_id != access.startup.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Transaction not found")
    return txn


@router.get("/transactions/{txn_id}/history", response_model=TransactionHistory)
def history(txn_id: uuid.UUID, access: Access = Depends(startup_access()), db: Session = Depends(get_db)) -> TransactionHistory:
    """Where a line came from (file, line number, fingerprint) and every change to its category since."""
    txn = _transaction(db, access, txn_id)
    imp = db.get(StatementImport, txn.import_id) if txn.import_id else None
    label = ALL_CATEGORIES.get
    events: list[HistoryEvent] = []
    if imp is not None:
        uploader = db.get(User, imp.uploaded_by) if imp.uploaded_by else None
        events.append(HistoryEvent(at=imp.created_at, by=uploader.full_name if uploader else None,
                                   what=f"Imported from {imp.filename}" + (f", line {txn.source_row}" if txn.source_row else "")))
    if txn.auto_source:
        first = f"Filed as {label(txn.auto_category, txn.auto_category)}" if txn.auto_category else "Left for review"
        events.append(HistoryEvent(at=txn.created_at, what=f"{first}: {txn.auto_reason}"))
    changes = db.execute(select(CategoryCorrection, User.full_name).outerjoin(User, User.id == CategoryCorrection.created_by)
                         .where(CategoryCorrection.transaction_id == txn.id).order_by(CategoryCorrection.created_at)).all()
    for c, name in changes:
        before = label(c.previous_category or "", "no category")
        after = label(c.new_category, c.new_category)
        events.append(HistoryEvent(at=c.created_at, by=name, what=f"Changed from {before} to {after}"
                                   if c.previous_category != c.new_category else f"Confirmed as {before}"))
    return TransactionHistory(transaction=TransactionOut.model_validate(txn), file=imp.filename if imp else None,
                              file_sha256=imp.file_sha256 if imp else None, statement_status=imp.balance_status if imp else None,
                              source_row=txn.source_row, first_category=txn.auto_category, first_source=txn.auto_source,
                              first_reason=txn.auto_reason, events=events)


@router.patch("/transactions/{txn_id}", response_model=CorrectionOut)
def correct(txn_id: uuid.UUID, body: CorrectionIn, request: Request, access: Access = Depends(startup_access()),
            db: Session = Depends(get_db)) -> CorrectionOut:
    txn = _transaction(db, access, txn_id)
    if body.category not in ALL_CATEGORIES:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Unknown category")
    before = txn.category
    others = service.correct(db, txn, body.category, access.user.id, body.remember)
    audit.record(db, "transaction.categorise", user_id=access.user.id, startup_id=access.startup.id, entity_id=txn.id,
                 details={"from": before, "to": body.category, "remembered": body.remember, "also_updated": others},
                 request=request)
    db.commit()
    db.refresh(txn)
    return CorrectionOut(transaction=TransactionOut.model_validate(txn), also_updated=others)
