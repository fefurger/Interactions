from bs4 import BeautifulSoup

def modify_html_file(html_string, output_file, edges_data):
    soup = BeautifulSoup(html_string, 'html.parser')
    update_style_tag(soup)
    modify_first_div(soup)
    adjust_network_div(soup)
    hide_config_div(soup)
    add_javascript(soup, edges_data)
    save_html_file(soup, output_file)


def update_style_tag(soup):
    style_tag = soup.find("style", string=lambda x: "#mynetwork" in x)
    if style_tag:
        styles = style_tag.string.split("}")
        styles = [style for style in styles if "height:" not in style]
        style_tag.string = "}".join(styles)


def modify_first_div(soup):
    first_div = soup.body.find("div")
    if first_div:
        first_div.attrs = {"style": "display:flex; justify-content: center; width: 100%; margin-top: 2em; padding-bottom: 2em; border-bottom: 1px solid black;"}
        insert_slider_div(first_div)


def insert_slider_div(parent_div):
    slider_div = BeautifulSoup("""
        <div>
            <br>
            <input type="range" id="thresholdSlider" min="0" max="1" step="0.02" value="0.2" oninput="updateEdges(this.value);">
            <label for="thresholdSlider">Threshold: <span id="thresholdLabel">0.2</span></label>
        </div>""", "html.parser")
    parent_div.insert(0, slider_div)


def adjust_network_div(soup):
    mynetwork_div = soup.find("div", {"id": "mynetwork"})
    if mynetwork_div:
        mynetwork_div.attrs.pop("class", None)
        mynetwork_div.extract()
        first_div = soup.body.find("div")
        first_div.insert_after(mynetwork_div)


def hide_config_div(soup):
    config_div = soup.find("div", {"id": "config"})
    if config_div:
        config_div.attrs["style"] = "display: none;"


def add_javascript(soup, edges_data):
    script_tag = soup.find("script")
    if script_tag:
        edges_data_js = f"var edgesData={str(edges_data)};"
        update_function_js = """
            function updateEdges(threshold) {
                document.getElementById("thresholdLabel").innerText = threshold;
                // Use the Vis.js API to update your edges based on the threshold
                edges.clear();
                for (var edge of edgesData) {
                    if (Math.abs(edge.weight) > threshold) {
                        edge.arrows = 'to';
                        if (edge.dashes==0) {
                            edge.dashes=false;
                        } else {
                        edge.dashes=true;
                        }
                        edges.add(edge);
                    }
                }
            }
        """
        script_tag.append(edges_data_js + update_function_js)


def save_html_file(soup, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(str(soup))
