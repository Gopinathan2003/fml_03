"""Build the required three-page OpenDocument Drawing report."""
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED

PAGES = [
    [
        ("title", "SPAM OR HAM"),
        ("subtitle", "From-scratch text classification with multinomial Naive Bayes"),
        ("head", "1. Task and data"),
        ("body", "The program reads every email*.txt file in the test folder in the current project directory. It prints 1 for spam and 0 for non-spam. The classifier is trained each time it runs, using the included labeled corpus."),
        ("body", "Training data: UCI SMS Spam Collection, compiled by Tiago A. Almeida and José María Gómez Hidalgo (2011). The original tab-separated file contains 5,574 English SMS messages labeled ham or spam. The repository record and citation are listed in DATASET_ATTRIBUTION.txt; the dataset is distributed under CC BY 4.0."),
        ("body", "This is a message dataset, not a full email corpus. It is included because the assignment permits publicly available data, and its labels make it suitable for a small text-classification exercise. SMS messages are shorter and have different formatting from email. That domain difference is a limitation: results on real emails can be weaker."),
        ("head", "2. Label mapping"),
        ("body", "The source labels are ham and spam. Ham means non-spam and is converted to output 0. Spam is converted to output 1, matching the assignment. The code reads the message body field; it does not use message headers."),
    ],
    [
        ("title", "FEATURES AND MODEL"),
        ("head", "3. Text features"),
        ("body", "Each message is converted to a bag of word tokens. Tokenization lowercases the text and keeps sequences of English letters (including a simple apostrophe contraction) and digit sequences. Punctuation is discarded. Repeated words are counted repeatedly. There is no stemming, stop-word removal, or feature selection."),
        ("body", "The vocabulary is the set of tokens seen in training. For a test message, tokens absent from that vocabulary are ignored. The resulting features are token counts, so word order is not represented."),
        ("head", "4. Multinomial Naive Bayes"),
        ("body", "For each class c in {ham, spam}, training counts the number of documents N_c and the number of occurrences n(w,c) of each vocabulary word w. The class prior is P(c) = N_c / N. With add-one smoothing (alpha = 1), the word probability is P(w|c) = (n(w,c) + 1) / (T_c + |V|), where T_c is the total number of training tokens in class c and |V| is vocabulary size."),
        ("body", "For a test message with token counts x_w, the program computes log P(c) + sum_w x_w log P(w|c). It predicts the class with the larger score. Logarithms avoid numerical underflow. If scores tie, the implementation predicts ham. This is multinomial Naive Bayes: it assumes token occurrences are conditionally independent given the class."),
        ("head", "5. Hyperparameters"),
        ("body", "The sole smoothing parameter is alpha = 1.0 (Laplace/add-one smoothing). No kernel is used. No tuning is performed: the value is fixed in classifier.py so the submission is reproducible and the method remains easy to inspect."),
    ],
    [
        ("title", "RUNNING AND LIMITATIONS"),
        ("head", "6. Training and prediction procedure"),
        ("body", "Run python3 classifier.py from this directory. The program reads data/SMSSpamCollection, builds counts and vocabulary from scratch, then reads files matching test/email*.txt. It prints one line per file with the filename and the assignment label. It uses only Python's standard library; no machine-learning package is used."),
        ("body", "The file email.txt is processed first when present; remaining matching files are listed by name. If no matching files exist, the program prints a short message. Put the training file at the supplied relative path and keep the test folder beside classifier.py."),
        ("head", "7. Evaluation and limitations"),
        ("body", "The assignment did not provide a held-out email test set, so this report does not claim a measured test accuracy. The instructor's test emails are the relevant evaluation set. A meaningful validation score should use a held-out, stratified split of this SMS corpus, with the split fixed before comparing model settings; it would still estimate SMS performance rather than email performance."),
        ("body", "Other limits are the independence assumption, the small token representation, and the difference between SMS and email. The model does not inspect URLs, sender addresses, HTML, or headers as separate signals. These are deliberate scope choices for a transparent from-scratch baseline."),
        ("head", "References"),
        ("body", "Almeida, T. A. and Hidalgo, J. M. G. (2011). SMS Spam Collection. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84. Dataset page: https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection"),
        ("body", "Student details: see Details.txt. Replace the name and roll-number placeholders before submission."),
    ],
]

NS = {
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "style": "urn:oasis:names:tc:opendocument:xmlns:style:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
    "draw": "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0",
    "fo": "urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0",
    "svg": "urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0",
}
for prefix, uri in NS.items():
    __import__("xml.etree.ElementTree", fromlist=["register_namespace"]).register_namespace(prefix, uri)
import xml.etree.ElementTree as ET

def q(prefix, name):
    return f"{{{NS[prefix]}}}{name}"

def make_content():
    root = ET.Element(q("office", "document-content"), {q("office", "version"): "1.2"})
    automatic = ET.SubElement(root, q("office", "automatic-styles"))
    for name, size, weight, after in [("Title", "22pt", "bold", "0.12in"), ("Subtitle", "12pt", "normal", "0.3in"), ("Heading", "15pt", "bold", "0.12in"), ("Body", "10pt", "normal", "0.14in")]:
        st = ET.SubElement(automatic, q("style", "style"), {q("style", "name"): name, q("style", "family"): "paragraph"})
        props = {q("fo", "font-size"): size, q("fo", "font-weight"): weight, q("fo", "margin-bottom"): after, q("fo", "font-family"): "Liberation Sans", q("fo", "line-height"): "120%"}
        if name == "Title":
            props[q("fo", "color")] = "#17365D"
        ET.SubElement(st, q("style", "paragraph-properties"), {q("fo", "keep-with-next"): "always"} if name in {"Title", "Subtitle", "Heading"} else {})
        ET.SubElement(st, q("style", "text-properties"), props)
    body = ET.SubElement(root, q("office", "body"))
    drawing = ET.SubElement(body, q("office", "drawing"))
    for page_number, page in enumerate(PAGES, 1):
        pg = ET.SubElement(drawing, q("draw", "page"), {q("draw", "name"): f"Page {page_number}", q("draw", "master-page-name"): "Default"})
        frame = ET.SubElement(pg, q("draw", "frame"), {q("draw", "name"): f"Report text {page_number}", q("draw", "style-name"): "Frame", q("svg", "x"): "0.65in", q("svg", "y"): "0.55in", q("svg", "width"): "7.0in", q("svg", "height"): "10.5in"})
        box = ET.SubElement(frame, q("draw", "text-box"))
        for kind, content in page:
            style = {"title": "Title", "subtitle": "Subtitle", "head": "Heading", "body": "Body"}[kind]
            p = ET.SubElement(box, q("text", "p"), {q("text", "style-name"): style})
            p.text = content
    return ET.tostring(root, encoding="UTF-8", xml_declaration=True)

def make_styles():
    root = ET.Element(q("office", "document-styles"), {q("office", "version"): "1.2"})
    auto = ET.SubElement(root, q("office", "automatic-styles"))
    layout = ET.SubElement(auto, q("style", "page-layout"), {q("style", "name"): "PageLayout"})
    ET.SubElement(layout, q("style", "page-layout-properties"), {q("fo", "page-width"): "8.27in", q("fo", "page-height"): "11.69in", q("fo", "margin-top"): "0in", q("fo", "margin-bottom"): "0in", q("fo", "margin-left"): "0in", q("fo", "margin-right"): "0in"})
    frame = ET.SubElement(auto, q("style", "style"), {q("style", "name"): "Frame", q("style", "family"): "graphic"})
    ET.SubElement(frame, q("style", "graphic-properties"), {q("draw", "textarea-vertical-align"): "top", q("fo", "padding"): "0in", q("draw", "fill"): "none", q("draw", "stroke"): "none"})
    masters = ET.SubElement(root, q("office", "master-styles"))
    master = ET.SubElement(masters, q("style", "master-page"), {q("style", "name"): "Default", q("style", "page-layout-name"): "PageLayout"})
    return ET.tostring(root, encoding="UTF-8", xml_declaration=True)

def main():
    with ZipFile("Report.odg", "w") as odg:
        odg.writestr("mimetype", "application/vnd.oasis.opendocument.graphics", compress_type=ZIP_STORED)
        odg.writestr("content.xml", make_content(), compress_type=ZIP_DEFLATED)
        odg.writestr("styles.xml", make_styles(), compress_type=ZIP_DEFLATED)
        odg.writestr("meta.xml", "<?xml version='1.0' encoding='UTF-8'?><office:document-meta xmlns:office='urn:oasis:names:tc:opendocument:xmlns:office:1.0' office:version='1.2'><office:meta/></office:document-meta>", compress_type=ZIP_DEFLATED)
        odg.writestr("META-INF/manifest.xml", "<?xml version='1.0' encoding='UTF-8'?><manifest:manifest xmlns:manifest='urn:oasis:names:tc:opendocument:xmlns:manifest:1.0' manifest:version='1.2'><manifest:file-entry manifest:full-path='/' manifest:media-type='application/vnd.oasis.opendocument.graphics'/><manifest:file-entry manifest:full-path='content.xml' manifest:media-type='text/xml'/><manifest:file-entry manifest:full-path='styles.xml' manifest:media-type='text/xml'/><manifest:file-entry manifest:full-path='meta.xml' manifest:media-type='text/xml'/></manifest:manifest>", compress_type=ZIP_DEFLATED)

if __name__ == "__main__":
    main()
