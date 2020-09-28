# Getting started
## 顔ベクトル計算前処理

* AnimeCVのインストール

    ```
    pip install git+https://github.com/kosuke1701/AnimeCV.git
    ```
* イラストからの顔検知およびembedding計算

    ```
    python encode_face_vectors.py --target-fn <TARGET> --image-root <ROOT_DIR> --cuda
    ```

    * `<TARGET>`は1列目がイラストIDで、2列目が`<ROOT_DIR>`からのイラスト画像の相対パスであるようなTSVファイル。
    * embeddingを保存したデータベースファイル`vectors.sql`が作成される。
    * データベース中にイラストIDが既存の画像は処理しないので、`<TARGET>`が更新された場合は単純に上記コマンドを再度実行すればよい。
    
## pixiv顔検索

[Notebookを開く](https://colab.research.google.com/github/kosuke1701/illust-search/blob/master/face_search.ipynb)
