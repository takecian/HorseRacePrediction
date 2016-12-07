# coding: utf-8
import numpy as np
import csv
import pickle as P
import sys

class DataGenerator:
  def __init__ (self) :
    self.files = ["Data/2007.csv", "Data/2008.csv", "Data/2009.csv", "Data/2010.csv", "Data/2011.csv", "Data/2012.csv", "Data/2013.csv", "Data/2014.csv", "Data/2015.csv", "Data/2016.csv"]
    self.target_result_name = ""
    self.target_result_column = 14 # 着順（答え）

    # self.target_race_colums = {1, 2, 4, 5, 6, 7, 8, 9} # 学習の要素
    self.target_race_colums = {3, 4, 6, 7, 8, 9} # 学習の要素ーレース情報
    # self.target_horse_colums = {11, 12, 14, 15, 17, 21, 22, 23, 24, 25} # 学習の要素ー馬の情報
    self.target_horse_colums = {10, 11, 12, 13, 15, 16, 18, 19, 20, 21, 22, 24, 26} # 学習の要素ー馬の情報
    # self.excluded_colums = {3, 13, 16, 18, 19, 26, 27, 28} # 学習には使わない要素

    self.train_data_dimension = len(self.target_race_colums) + len(self.target_horse_colums) * 18 # 1レース18頭立てとして計算する
    print("train_data_dimension = {}".format(self.train_data_dimension))

    self.train_data = np.array([], dtype=np.float32).reshape(0, self.train_data_dimension)
    self.train_data_answer = np.array([], dtype=np.int32)
    # self.train_data_answer = np.array([], dtype=np.int32).reshape(0, 18)

    # テスト対象
    self.test_date = '0912'#'1610-中山' # この条件にマッチするレースを検証データとする '0712-阪神'# 
    self.test_data = np.array([], dtype=np.float32).reshape(0, self.train_data_dimension)
    self.test_data_answer = np.array([], dtype=np.int32)
    # self.test_data_answer = np.array([], dtype=np.int32).reshape(0, 18)

    # 学習用に数値に変換する
    self.dataMap = {
      3 : { "札幌": 0, "函館": 1, "福島": 2, "東京": 3, "中山": 4, "京都": 5, "新潟": 6, "阪神": 7, "中京": 8, "小倉": 9 },
      6 :  { "芝" : 0, "ダ" : 1 },
      9 :  { "不" : 0,  "重" : 1, "稍" : 2, "良" : 3 },
      15 : {"牡" : 0, "牝" : 1, "セ" : 2},
      18 : {"逃げ" : 0, "先行" : 1, "中団" : 2, "差し" : 3, "後方" : 4, "追込" : 5, "ﾏｸﾘ" : 6, "" : 7}
    }

  def read(self):
    hurdle_race_count = 0
    header = []

    for file in self.files:
      print("file name = {}".format(file))
      with open(file, "r") as f:
        reader = csv.reader(f)

        previous_race_id = ""

        race_parameter = np.array([], dtype=np.float32)
        horses_parameter = np.array([], dtype=np.float32).reshape(0, len(self.target_horse_colums) + 1)

        # 障害は除くデータで予測データを作成
        answer = np.int32(-1)
        for idx, row in enumerate(reader):
          # if idx == 0: # skip header
          #   continue
          # elif idx == 1: # skip header
          #   for i, col in enumerate(row):
          #     # if i in self.excluded_colums:
          #     #   print("index = {}, name = {}, excluded from training".format(i, row[i]))
          #     if i in self.target_race_colums:
          #       print("index = {}, name = {}, used for training - race".format(i, row[i]))
          #     if i in self.target_horse_colums:
          #       print("index = {}, name = {}, used for training - horse".format(i, row[i]))
          #     header = row
          #   continue
          # elif row[4] == '障害' :
          #   hurdle_race_count += 1
          #   continue
          
          current_race_id = "{}{}{}-{}-{}".format(row[0], row[1], row[2], row[3], row[4]).replace(u'\ufeff', '')
          current_date = "{}{}".format(row[0], row[1])
    		
          if len(previous_race_id) > 0 and previous_race_id != current_race_id:
            sys.stdout.write("\rrace_id = {}".format(previous_race_id))
            sys.stdout.flush()

            # output previous race info
            # print("race_parameter = {}".format(race_parameter))
            # print("horses_parameter = {}".format(horses_parameter))

            horses, race_answer = np.hsplit(horses_parameter, [len(self.target_horse_colums)]) # 馬の情報と着順を分離
            rase_all_parameter = np.hstack((race_parameter, horses.flatten())) # レースと全馬の情報を結合
            rase_all_parameter = np.pad(rase_all_parameter, (0, self.train_data_dimension - len(rase_all_parameter)), 'constant', constant_values=(0, -1)) # 頭数がすくない場合padding

            race_answer = np.pad(race_answer.flatten(), (0, 18 - len(race_answer.flatten())), 'constant', constant_values=(0, 0)) # 頭数がすくない場合padding
            race_answer = race_answer.astype(np.int32)
            # print("race_answer = {}".format(race_answer))
            # return
            # print("loader.train_data = {}, horses_parameter = {}, rase_all_parameter = {}".format(self.train_data.dtype,horses_parameter.dtype, rase_all_parameter.dtype))
            # print("loader.train_data_answer = {}".format(self.train_data_answer))
            # print("loader.test_data = {}".format(self.test_data.dtype))
            # print("loader.test_data_answer = {}".format(self.test_data_answer))
            if current_date != self.test_date:
              self.train_data = np.vstack([self.train_data, rase_all_parameter])
              # self.train_data_answer = np.vstack([self.train_data_answer, race_answer])
              # self.train_data_answer = np.vstack([self.train_data_answer, race_answer])
              self.train_data_answer = np.append(self.train_data_answer, np.int32(answer))
            else:
              self.test_data = np.vstack([self.test_data, rase_all_parameter])
              # self.test_data_answer = np.vstack([self.test_data_answer, race_answer])
              self.test_data_answer = np.append(self.test_data_answer, np.int32(answer))
            # initialize 
            horses_parameter = np.array([], dtype=np.float32).reshape(0, len(self.target_horse_colums) + 1)
          else:
            # print("current race = {}".format(current_race_id))
            pass
            
          # race_parameter_label = []
          # horse_parameter_label = []
          race_parameter = np.array([], dtype=np.float32)
          horse_parameter = np.array([], dtype=np.float32)
          
          # if current_race_id == '070811-札幌-8': # "080816-札幌-1": #070901-札幌-1":
          #   return
          # マスタデータで数値化
          for i, col in enumerate(row): # 馬一頭分の情報の詳細
            # if i == 0: 
              # if self.test_row_no == -1 and col == self.test_date :
              #   self.test_row_no = (idx - hurdle_race_count)
              # race_parameter_label.append(header[i])
              # race_parameter = np.append(race_parameter, col.replace('-',''))
            if i == self.target_result_column: 
              # self.target_result_name = header[i]
              #answer = int(col)  # 正解フラグを立てるならここをいじる。3以内の馬にフラグをたてる
              # answer = 1 if int(col) == 1 else 0  # 正解フラグを立てるならここをいじる。1位の馬にフラグをたてる
              if int(col) == 1:
                answer = int(row[11]) - 1
                # print("answer = {}".format(answer))
            elif i in self.target_race_colums:
              # race_parameter_label.append(header[i])
              if i in self.dataMap :
                race_parameter = np.append(race_parameter, np.float32(self.dataMap[i][col]))
              else:
                val = np.float32(col) if col else np.float32(0)
                race_parameter = np.append(race_parameter, val)
            elif i in self.target_horse_colums:
              # horse_parameter_label.append(header[i])
              if i in self.dataMap :
                horse_parameter = np.append(horse_parameter, np.float32(self.dataMap[i][col]))
              else:
                val = np.float32(col) if col else np.float32(0) # 発走除外の馬などは空欄になる項目があるのでその場合は0で埋める
                horse_parameter = np.append(horse_parameter, val)
          if len(horse_parameter) == 0:
            print("horse_parameter empty")
            break
          # print("horse_parameter = {}, answer = {}".format(horse_parameter.dtype, answer))
          # return
          horse_parameter = np.append(horse_parameter, np.float32(answer))
          horses_parameter = np.vstack([horses_parameter, horse_parameter])
          previous_race_id = current_race_id


def setup():
  loader = DataGenerator()
  loader.read()

  with open('train_data.pickle', 'wb') as f:
    P.dump(loader.train_data, f)

  with open('train_data_answer.pickle', 'wb') as f:
    P.dump(loader.train_data_answer, f)

  with open('test_data.pickle', 'wb') as f:
    P.dump(loader.test_data, f)

  with open('test_data_answer.pickle', 'wb') as f:
    P.dump(loader.test_data_answer, f)

  print("train_data count = {}".format(len(loader.train_data)))
  print("train_data_answer count = {}".format(len(loader.train_data_answer)))
  print("test_data count = {}".format(len(loader.test_data)))
  print("test_data_answer count = {}".format(len(loader.test_data_answer)))
  print("loader.train_data = {}, shape = {}".format(loader.train_data.dtype, loader.train_data.shape))
  print("loader.train_data_answer = {}, shape = {}".format(loader.train_data_answer.dtype, loader.train_data_answer.shape))
  print("loader.test_data = {}, shape = {}".format(loader.test_data.dtype, loader.test_data.shape))
  print("loader.test_data_answer = {}, shape = {}".format(loader.test_data_answer.dtype, loader.test_data_answer.shape))

if __name__ == '__main__':
    setup()
