//! Font-decoding guard for the PDF text extractor (issue #617).
//!
//! A font resource name such as `/F1` is scoped to one resource dictionary, so two
//! pages (or a page and a form XObject drawn on it) may both call their font `/F1`
//! while pointing at different font objects. pdf-extract 0.12.0 caches fonts by that
//! local name for the whole document (upstream commit 205c16ff, reported as
//! jrmuizel/pdf-extract#157): every later `/F1` is decoded with the first font's
//! encoding, and extraction returns plausible-looking but wrong text as `Ok`.
//! Nothing downstream can tell once that text is normalized, indexed, and stored.
//!
//! The fixtures pair Helvetica with Symbol. Symbol's built-in encoding maps the bytes
//! `abg` to `αβγ`, so text decoded with the wrong font shows up as Latin letters
//! where Greek ones belong. `pdf_ordered_text.rs` cannot see this: it checks reading
//! order on two real papers whose sentences decode the same either way.

use lopdf::content::{Content, Operation};
use lopdf::{dictionary, Dictionary, Document, Object, ObjectId, Stream};
use wenlan_core::sources::directory::extract_pdf_text;

/// A base-14 font dictionary; no `/Encoding`, so the font's built-in one applies.
fn base14_font(doc: &mut Document, base_font: &str) -> ObjectId {
    doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => base_font,
    })
}

/// Content-stream operations that draw `text` with whatever `/F1` resolves to.
fn draw_with_f1(text: &str) -> Vec<Operation> {
    vec![
        Operation::new("BT", vec![]),
        Operation::new("Tf", vec!["F1".into(), 24.into()]),
        Operation::new("Td", vec![20.into(), 100.into()]),
        Operation::new("Tj", vec![Object::string_literal(text)]),
        Operation::new("ET", vec![]),
    ]
}

/// Assemble a PDF with one page per (resources, operations) pair. Generated with
/// lopdf so the xref and trailer are byte-correct, as the other PDF fixtures are.
fn build_pdf(doc: &mut Document, pages: Vec<(Dictionary, Vec<Operation>)>) -> Vec<u8> {
    let pages_id = doc.new_object_id();
    let mut kids = Vec::new();
    for (resources, operations) in pages {
        let resources_id = doc.add_object(resources);
        let content = Content { operations }.encode().unwrap();
        let content_id = doc.add_object(Stream::new(dictionary! {}, content));
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "Contents" => content_id,
            "Resources" => resources_id,
            "MediaBox" => vec![0.into(), 0.into(), 300.into(), 144.into()],
        });
        kids.push(Object::Reference(page_id));
    }
    let count = kids.len() as i64;
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => kids,
            "Count" => count,
        }),
    );
    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    doc.trailer.set("Root", catalog_id);

    let mut buf = Vec::new();
    doc.save_to(&mut buf).unwrap();
    buf
}

/// Page 1 draws "Hello" with Helvetica; page 2 draws the bytes "abg" with Symbol.
/// Both pages name their font `/F1`.
#[test]
fn pdf_text_decodes_each_page_with_its_own_font() {
    let mut doc = Document::with_version("1.5");
    let helvetica = base14_font(&mut doc, "Helvetica");
    let symbol = base14_font(&mut doc, "Symbol");
    let bytes = build_pdf(
        &mut doc,
        vec![
            (
                dictionary! { "Font" => dictionary! { "F1" => helvetica } },
                draw_with_f1("Hello"),
            ),
            (
                dictionary! { "Font" => dictionary! { "F1" => symbol } },
                draw_with_f1("abg"),
            ),
        ],
    );

    let text = extract_pdf_text(&bytes).expect("pdf extraction");
    assert!(
        text.contains("Hello"),
        "page 1 (Helvetica) missing: {text:?}"
    );
    assert!(
        text.contains("αβγ") && !text.contains("abg"),
        "page 2's /F1 is Symbol, so its bytes must decode as Greek letters; Latin ones \
         mean page 1's font was reused for page 2 (pdf-extract#157): {text:?}"
    );
}

/// One page draws "Hello" with its own `/F1` (Helvetica), then `Do`es a form XObject
/// whose own resources map `/F1` to Symbol and which draws the bytes "abg".
#[test]
fn pdf_text_decodes_form_xobject_with_its_own_font() {
    let mut doc = Document::with_version("1.5");
    let helvetica = base14_font(&mut doc, "Helvetica");
    let symbol = base14_font(&mut doc, "Symbol");
    let form = doc.add_object(Stream::new(
        dictionary! {
            "Type" => "XObject",
            "Subtype" => "Form",
            "BBox" => vec![0.into(), 0.into(), 300.into(), 144.into()],
            "Resources" => dictionary! { "Font" => dictionary! { "F1" => symbol } },
        },
        Content {
            operations: draw_with_f1("abg"),
        }
        .encode()
        .unwrap(),
    ));
    let mut operations = draw_with_f1("Hello");
    operations.push(Operation::new("Do", vec!["Fig".into()]));
    let bytes = build_pdf(
        &mut doc,
        vec![(
            dictionary! {
                "Font" => dictionary! { "F1" => helvetica },
                "XObject" => dictionary! { "Fig" => form },
            },
            operations,
        )],
    );

    let text = extract_pdf_text(&bytes).expect("pdf extraction");
    assert!(
        text.contains("Hello"),
        "page text (Helvetica) missing: {text:?}"
    );
    assert!(
        text.contains("αβγ") && !text.contains("abg"),
        "the form's /F1 is Symbol, so its bytes must decode as Greek letters; Latin ones \
         mean the page's font was reused inside the form (pdf-extract#157): {text:?}"
    );
}
