"""Built-in field schemas per document category for key-field extraction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSpec:
    """A single extractable field.

    ``type`` drives value parsing/validation: text | date | money | email | phone.
    ``labels`` are case-insensitive line-label synonyms used by the rule engine
    (e.g. "Invoice Number:", "Invoice No:", "Invoice #").
    """

    name: str
    type: str = "text"
    labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class Schema:
    category: str
    fields: tuple[FieldSpec, ...]

    @property
    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]


INVOICE = Schema(
    "invoice",
    (
        FieldSpec("invoice_number", "text", ("invoice number", "invoice no", "invoice #", "number")),
        FieldSpec("invoice_date", "date", ("invoice date", "date of invoice", "issue date", "date")),
        FieldSpec("due_date", "date", ("due date", "payment due", "date due", "pay by")),
        FieldSpec("total_amount", "money", ("total", "amount due", "total due", "grand total", "amount")),
        FieldSpec("subtotal", "money", ("subtotal", "net amount", "amount before tax")),
        FieldSpec("tax_amount", "money", ("tax", "vat", "gst", "sales tax", "tax amount")),
        FieldSpec("currency", "text", ("currency",)),
        FieldSpec("vendor", "text", ("from", "bill from", "vendor", "supplier", "seller", "remittance to")),
        FieldSpec("bill_to", "text", ("bill to", "billed to", "customer", "client")),
        FieldSpec("payment_terms", "text", ("payment terms", "terms")),
    ),
)

RECEIPT = Schema(
    "receipt",
    (
        FieldSpec("receipt_number", "text", ("receipt number", "receipt no", "receipt #", "transaction id")),
        FieldSpec("date", "date", ("date", "transaction date")),
        FieldSpec("total_amount", "money", ("total", "amount paid", "amount", "grand total")),
        FieldSpec("merchant", "text", ("merchant", "store", "vendor", "seller", "paid to")),
        FieldSpec("payment_method", "text", ("payment method", "paid via", "card")),
    ),
)

PURCHASE_ORDER = Schema(
    "purchase_order",
    (
        FieldSpec("po_number", "text", ("po number", "purchase order number", "purchase order no", "po no", "po #")),
        FieldSpec("order_date", "date", ("order date", "po date", "date")),
        FieldSpec("delivery_date", "date", ("delivery date", "required by", "ship by", "needed by")),
        FieldSpec("total_amount", "money", ("total", "amount", "grand total", "order total")),
        FieldSpec("vendor", "text", ("vendor", "supplier", "seller", "ship to")),
        FieldSpec("bill_to", "text", ("bill to", "buyer", "customer")),
    ),
)

QUOTATION = Schema(
    "quotation",
    (
        FieldSpec("quote_number", "text", ("quote number", "quotation number", "quote no", "quotation no", "quote #")),
        FieldSpec("quote_date", "date", ("quote date", "quotation date", "date")),
        FieldSpec("valid_until", "date", ("valid until", "valid through", "expiry date", "expires", "offer valid until")),
        FieldSpec("total_amount", "money", ("total", "quoted amount", "grand total", "amount")),
        FieldSpec("vendor", "text", ("from", "vendor", "supplier", "seller")),
        FieldSpec("customer", "text", ("to", "customer", "client", "prepared for")),
    ),
)

CONTRACT = Schema(
    "contract",
    (
        FieldSpec("parties", "text", ("between", "parties", "by and between")),
        FieldSpec("effective_date", "date", ("effective date", "commencement date", "start date", "date of agreement")),
        FieldSpec("expiration_date", "date", ("expiration date", "end date", "termination date", "expiry date")),
        FieldSpec("governing_law", "text", ("governing law", "jurisdiction", "governed by")),
        FieldSpec("contract_value", "money", ("contract value", "total value", "fee", "compensation", "amount")),
    ),
)

RESUME = Schema(
    "resume",
    (
        FieldSpec("name", "text", ("name",)),
        FieldSpec("email", "email", ("email", "e-mail")),
        FieldSpec("phone", "phone", ("phone", "telephone", "mobile", "cell")),
        FieldSpec("location", "text", ("location", "address", "city")),
        FieldSpec("current_title", "text", ("current title", "title", "headline", "position")),
    ),
)

# Fallback schema for categories without a dedicated one: generic contact/date fields.
GENERIC = Schema(
    "other",
    (
        FieldSpec("title", "text", ("title", "subject", "re")),
        FieldSpec("date", "date", ("date",)),
        FieldSpec("author", "text", ("author", "from", "prepared by", "written by")),
        FieldSpec("email", "email", ("email", "e-mail")),
        FieldSpec("phone", "phone", ("phone", "telephone", "mobile")),
        FieldSpec("reference_number", "text", ("reference number", "reference no", "ref no", "ref #", "document id")),
    ),
)

SCHEMAS: dict[str, Schema] = {
    s.category: s
    for s in (INVOICE, RECEIPT, PURCHASE_ORDER, QUOTATION, CONTRACT, RESUME)
}


def schema_for(category: str) -> Schema:
    """Return the built-in schema for a category, or the generic one."""
    return SCHEMAS.get(category, GENERIC)
