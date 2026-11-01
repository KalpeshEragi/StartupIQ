# Expense categoriser: check on Indian bank narrations

The categoriser is trained on UK expense claims. This check asks whether it still works on the narration format of
Indian business current-account statements (AgamiAI/Indian-Bank-Statements, Apache-2.0, synthetic).

## Method

1. All 32,349 debit and credit lines from the 200 digital statements go through the bank-format rules
   (`backend/app/categorisation/formats.py`). Rules recognise every structural line: customer receipts, GST and TDS
   payments, EMIs, bank charges, cash, reversals, failed lines and payments to named people.
2. The 638 lines left are expense lines with a stated purpose (`NEFT Dr-...-RENT PAYMENT-MONTHLY RENT`,
   `...-SOFTWARE SUBSCRIPTION`, `...-CLOUD SERVICES`, `...-UTILITY PAYMENT`, `...-INSURANCE PREMIUM` and similar).
   Their expected category follows from the stated purpose.
3. The installed model (Linear SVM, version 20261101-b1c3c47f) predicts each of them.

## Result

**611 of 638 correct (95.8%).** Every wrong prediction fell below the confidence threshold, so in the product those
27 lines would go to the founder's review inbox rather than being filed wrongly.

| Narration (cleaned) | Lines | Expected | Predicted | Filed automatically |
| --- | --- | --- | --- | --- |
| rent payment monthly rent | 196 | Rent & premises | Rent & premises | yes |
| software subscription | 107 | Software & IT | Software & IT | yes |
| cloud services | 73 | Software & IT | Software & IT | yes |
| utility payment | 67 | Utilities | Utilities | yes |
| insurance premium | 64 | Insurance | Insurance | yes |
| insurance subscription | 49 | Insurance | Insurance | yes |
| cloud subscription | 44 | Software & IT | Software & IT | yes |
| utility subscription | 27 | Utilities | Software & IT | no (sent to review) |
| utility subscription | 11 | Utilities | Utilities | no (sent to review) |

## Limits

These narrations come from a small set of templates, so this checks the format transfer, not breadth of vendors.
Indian vendor names (Razorpay, Zoho, Swiggy...) are learned per startup through the founder's corrections.
