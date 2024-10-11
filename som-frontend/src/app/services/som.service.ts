import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class SomService {
  private apiUrl = 'http://localhost:5000';

  constructor(private http: HttpClient) { }

  trainSom(): Observable<any> {
    return this.http.post(`${this.apiUrl}/train-som`, {});
  }

  getImageManual(): Observable<any> {
    return this.http.get(`${this.apiUrl}/get-image/manual`, { responseType: 'blob' });
  }

  getImageMinisom(): Observable<any> {
    return this.http.get(`${this.apiUrl}/get-image/minisom`, { responseType: 'blob' });
  }

  getAccuracy(somType: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/get-accuracy/${somType}`);
  }
}
