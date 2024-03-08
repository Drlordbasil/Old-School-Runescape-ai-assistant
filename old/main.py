import os
import requests
import json
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
from kivy.utils import get_color_from_hex
from kivy.properties import StringProperty
from threading import Thread
from kivy.core.clipboard import Clipboard

class OSRSGrandExchangeApp(App):
    suggestions_text = StringProperty("")

    def copy_to_clipboard(self, instance):
        text_to_copy = f"Item Suggestions:\n{self.suggestions_text}"
        Clipboard.copy(text_to_copy)

    def build(self):
        self.title = "OSRS Grand Exchange Helper"
        layout = BoxLayout(orientation="vertical", spacing=10, padding=20)
        layout.bind(size=self._update_layout)

        header = Label(text="OSRS Grand Exchange Helper", size_hint=(1, None), height=50, font_size=24, bold=True, color=get_color_from_hex("#FFFFFF"))
        header.bind(size=self._update_widget_size)
        layout.add_widget(header)

        input_layout = BoxLayout(orientation="horizontal", spacing=10, size_hint=(1, None), height=40)
        self.starting_gold_input = TextInput(text="10000", multiline=False, size_hint=(0.7, None), height=40, background_color=get_color_from_hex("#FFFFFF"), foreground_color=get_color_from_hex("#000000"))
        input_layout.add_widget(Label(text="Starting Gold:", size_hint=(0.3, None), height=40, color=get_color_from_hex("#FFFFFF")))
        input_layout.add_widget(self.starting_gold_input)
        layout.add_widget(input_layout)

        button_layout = BoxLayout(orientation="horizontal", spacing=10, size_hint=(1, None), height=50)
        self.fetch_button = Button(text="Fetch Prices and Generate Suggestions", size_hint=(0.7, None), height=50, background_color=get_color_from_hex("#4CAF50"), color=get_color_from_hex("#FFFFFF"), bold=True)
        self.fetch_button.bind(on_press=self.fetch_prices_and_generate_suggestions)
        self.train_button = Button(text="Train Model", size_hint=(0.3, None), height=50, background_color=get_color_from_hex("#2196F3"), color=get_color_from_hex("#FFFFFF"), bold=True)
        self.train_button.bind(on_press=self.train_model)
        self.copy_button = Button(text="Copy to Clipboard", size_hint=(0.3, None), height=50, background_color=get_color_from_hex("#FFC107"), color=get_color_from_hex("#FFFFFF"), bold=True)
        self.copy_button.bind(on_press=self.copy_to_clipboard)
        button_layout.add_widget(self.fetch_button)
        button_layout.add_widget(self.train_button)
        button_layout.add_widget(self.copy_button)
        layout.add_widget(button_layout)

        scroll_view = ScrollView(size_hint=(1, 1))
        scroll_layout = BoxLayout(orientation="vertical", spacing=10, size_hint_y=None)
        scroll_layout.bind(minimum_height=scroll_layout.setter("height"))

        self.suggestions_label = Label(text="Item Suggestions:", size_hint_y=None, height=400, color=get_color_from_hex("#FFFFFF"), text_size=(Window.width - 40, None), halign="left", valign="top")
        scroll_layout.add_widget(self.suggestions_label)

        scroll_view.add_widget(scroll_layout)
        layout.add_widget(scroll_view)

        self.root = layout
        return layout

    def fetch_prices_and_generate_suggestions(self, instance):
        self.fetch_button.disabled = True
        starting_gold = int(self.starting_gold_input.text)
        thread = Thread(target=self.fetch_prices_and_generate_suggestions_thread, args=(starting_gold,))
        thread.start()

    def fetch_prices_and_generate_suggestions_thread(self, starting_gold):
        scraper = OSRSScraper()
        items_data = scraper.scrape_data()
        if items_data:
            suggestions = generate_item_suggestions(items_data, starting_gold)
            if suggestions:
                self.suggestions_text = f"Item Suggestions:\n{format_suggestions(suggestions)}"
            else:
                self.suggestions_text = "No item suggestions found."
        else:
            self.suggestions_text = "Error fetching item prices or item mapping."
        self.fetch_button.disabled = False

    def train_model(self, instance):
        self.train_button.disabled = True
        thread = Thread(target=self.train_model_thread)
        thread.start()

    def train_model_thread(self):
        scraper = OSRSScraper()
        items_data = scraper.scrape_data()
        if items_data:
            model_file = "model.pkl"
            if os.path.exists(model_file):
                with open(model_file, "rb") as file:
                    model = pickle.load(file)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)

            X, y = prepare_training_data(items_data)
            model.fit(X, y)

            with open(model_file, "wb") as file:
                pickle.dump(model, file)

            self.suggestions_text = "Model training completed."
        else:
            self.suggestions_text = "Error fetching item prices or item mapping."
        self.train_button.disabled = False

    def _update_layout(self, instance, size):
        instance.padding = (20, 20, 20, 20)

    def _update_widget_size(self, instance, size):
        instance.size_hint_y = None
        instance.height = size[1]

    def on_suggestions_text(self, instance, value):
        self.suggestions_label.text = value
        self.suggestions_label.texture_update()
        self.suggestions_label.height = self.suggestions_label.texture_size[1]

class OSRSScraper:
    def __init__(self):
        self.item_names = self.fetch_item_names()
        self.buy_limits = self.fetch_buy_limits()
        self.MIN_PROFIT = 3
        self.MIN_FLUCTUATION = 0
        self.MIN_ROI = 0
        self.MIN_SELL_VOLUME = 100
        self.MIN_BUY_VOLUME = 100
        self.api_url_latest = "https://prices.runescape.wiki/api/v1/osrs/latest"
        self.api_url_5m = "https://prices.runescape.wiki/api/v1/osrs/5m"

    def fetch_data(self, api_url):
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.json()['data']
        else:
            print("Failed to retrieve data from the RuneScape API")
            return None

    def fetch_item_names(self):
        mapping_url = "https://prices.runescape.wiki/api/v1/osrs/mapping"
        response = requests.get(mapping_url)
        item_names = {}
        if response.status_code == 200:
            mapping_data = response.json()
            for item in mapping_data:
                item_names[str(item['id'])] = item['name']
        return item_names

    def fetch_buy_limits(self):
        buy_limits_url = "https://prices.runescape.wiki/api/v1/osrs/mapping"
        response = requests.get(buy_limits_url)
        buy_limits = {}
        if response.status_code == 200:
            buy_limits_data = response.json()
            for item in buy_limits_data:
                buy_limits[str(item['id'])] = item.get('limit', 0)
        return buy_limits

    def scrape_data(self):
        data_latest = self.fetch_data(self.api_url_latest)
        data_5m = self.fetch_data(self.api_url_5m)
        items_data = []

        if data_latest and data_5m:
            for item_id, item_data_latest in data_latest.items():
                if item_id in data_5m:
                    item_data_5m = data_5m[item_id]
                    high_price = item_data_latest.get('high', 0)
                    low_price = item_data_latest.get('low', 0)
                    average_price_5m = item_data_5m.get('avgHighPrice', 0)
                    average_high_price = item_data_5m.get('avgHighPrice', 0)
                    average_low_price = item_data_5m.get('avgLowPrice', 0)
                    high_price_volume = item_data_5m.get('highPriceVolume', 0)
                    low_price_volume = item_data_5m.get('lowPriceVolume', 0)
                    buy_volume = low_price_volume
                    sell_volume = high_price_volume

                    high_price = average_high_price - average_high_price * 0.01 if average_high_price else 0
                    low_price = int(average_low_price * 0.99) if average_low_price else 0
                    average_price_5m = int(average_price_5m) if average_price_5m else 0

                    if high_price > 0 and low_price > 0:
                        potential_profit = high_price - low_price
                        profit_margin = (potential_profit / low_price) * 100

                        if average_price_5m > 0:
                            fluctuation = abs(high_price - average_price_5m) / average_price_5m
                            roi = potential_profit / average_price_5m

                            if (
                                profit_margin >= self.MIN_PROFIT
                                and fluctuation >= self.MIN_FLUCTUATION
                                and roi >= self.MIN_ROI
                                and sell_volume >= self.MIN_SELL_VOLUME
                                and buy_volume >= self.MIN_BUY_VOLUME
                            ):
                                item_name = self.item_names.get(item_id, "Unknown Item")
                                buy_limit = self.buy_limits.get(item_id, 0)
                                item_data = {
                                    "Item ID": item_id,
                                    "Item Name": item_name,
                                    "High (Sell)": high_price,
                                    "High Volume": high_price_volume,
                                    "Low (Buy)": low_price,
                                    "Low Volume": low_price_volume,
                                    "5-Minute Average High Price": average_price_5m,
                                    "ROI": roi,
                                    "Potential Profit": potential_profit,
                                    "Price Fluctuation": fluctuation * 100,
                                    "Buy Limit": buy_limit,
                                }
                                items_data.append(item_data)
        else:
            print("Failed to retrieve data from the RuneScape API")
            return None

        return items_data

def generate_item_suggestions(items_data, starting_gold):
    model_file = "model.pkl"
    if os.path.exists(model_file):
        with open(model_file, "rb") as file:
            model = pickle.load(file)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    X, _ = prepare_training_data(items_data)
    X_normalized = StandardScaler().fit_transform(X)
    predictions = model.predict(X_normalized)

    suggestions = []
    for i, item in enumerate(items_data):
        prediction = predictions[i]
        if prediction > 0:
            item["Predicted Profit"] = prediction
            buy_limit = item["Buy Limit"]
            buy_price = item["Low (Buy)"]
            max_quantity = min(buy_limit, starting_gold // buy_price)
            item["Max Quantity"] = max_quantity
            suggestions.append(item)

    suggestions.sort(key=lambda x: x["Predicted Profit"], reverse=True)
    return suggestions[:5]

def prepare_training_data(items_data):
    X = []
    y = []
    for item in items_data:
        X.append([
            item["High (Sell)"],
            item["Low (Buy)"],
            item["High Volume"],
            item["Low Volume"],
            item["5-Minute Average High Price"],
            item["Price Fluctuation"],
            item["Buy Limit"],
            item["ROI"]
        ])
        y.append(item["Potential Profit"])

    X_normalized = StandardScaler().fit_transform(X)
    y_log_transformed = np.log1p(y)
    return X_normalized, y_log_transformed

def format_suggestions(suggestions):
    formatted_suggestions = []
    for suggestion in suggestions:
        formatted_suggestion = f"- {suggestion['Item Name']}\n  Buy Price: {suggestion['Low (Buy)']}\n  Sell Price: {suggestion['High (Sell)']}\n  Potential Profit: {suggestion['Potential Profit']} per item\n  Buy Limit: {suggestion['Buy Limit']}\n  Max Quantity: {suggestion['Max Quantity']}\n"
        formatted_suggestions.append(formatted_suggestion)
    return "\n".join(formatted_suggestions)

if __name__ == "__main__":
    Window.clearcolor = get_color_from_hex("#1E1E1E")
    Window.size = (800, 600)
    OSRSGrandExchangeApp().run()
